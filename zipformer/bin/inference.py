#!/usr/bin/env python3
#
# Copyright 2021-2026 Xiaomi Corporation (Author: Wei Kang,
#                                                 Zengwei Yao,
#                                                 Fangjun Kuang,
#                                                 Zengrui Jin)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified inference script for Zipformer models.

Supports JIT and ONNX models, streaming and non-streaming,
transducer and CTC heads. Uses greedy search decoding.

Usage examples:

(1) JIT non-streaming transducer:

  python inference.py \
    --model-type jit \
    --model ./exp/jit_model.pt \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(2) JIT streaming transducer:

  python inference.py \
    --model-type jit --streaming true \
    --model ./exp/jit_script_chunk_16_left_128.pt \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(3) ONNX non-streaming transducer:

  python inference.py \
    --model-type onnx \
    --encoder ./exp/encoder.onnx \
    --decoder ./exp/decoder.onnx \
    --joiner ./exp/joiner.onnx \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(4) ONNX non-streaming CTC:

  python inference.py \
    --model-type onnx --ctc true \
    --model ./exp/model.onnx \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(5) ONNX streaming transducer:

  python inference.py \
    --model-type onnx --streaming true \
    --encoder ./exp/encoder-streaming.onnx \
    --decoder ./exp/decoder-streaming.onnx \
    --joiner ./exp/joiner-streaming.onnx \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(6) ONNX streaming CTC:

  python inference.py \
    --model-type onnx --streaming true --ctc true \
    --model ./exp/ctc-streaming.onnx \
    --tokens ./data/lang_bpe_500/tokens.txt \
    /path/to/foo.wav /path/to/bar.wav

(7) Download model from HuggingFace (JIT):

  python inference.py \
    --model-type jit \
    --hf-model pkufool/zipformer-medium \
    /path/to/foo.wav /path/to/bar.wav

(8) Download model from HuggingFace (ONNX transducer):

  python inference.py \
    --model-type onnx \
    --hf-model pkufool/zipformer-medium \
    /path/to/foo.wav /path/to/bar.wav
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import List

import torch
import torchaudio

from zipformer.utils import (
    str2bool,
    SymbolTable,
    AttributeDict,
    token_ids_to_text,
    stack_states,
    unstack_states,
)
from zipformer.decode.search import (
    greedy_search,
    streaming_greedy_search,
    ctc_greedy_search,
    streaming_ctc_greedy_search,
)
from zipformer.decode.stream import DecodeStream
from zipformer.modules.model import (
    OnnxCtcModel,
    OnnxStreamingCtcModel,
    OnnxTransducerModel,
    OnnxStreamingTransducerModel,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["jit", "onnx"],
        help="Model format: 'jit' for TorchScript, 'onnx' for ONNX.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="""
        Data type for the model: 'float32', 'float16', or 'int8'. Only used when
        --ms-model or --hf-model is specified to determine which model file to download.
        For --model-type jit or onnx with user-provided model files, the data type is
        determined by the model file itself and this argument is ignored.
        """,
    )

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=False,
        help="Whether to use streaming inference.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="""
        Chunk size for streaming inference. Only used when --ms-model or --hf-model is specified,
        to determine which model file to download. Ignored if --model-type is jit or onnx with
        user-provided model files.
        """,
    )

    parser.add_argument(
        "--left-context-frames",
        type=int,
        default=128,
        help="""
        Number of left context frames for streaming inference (after encoder embedding).
        Only used when --ms-model or --hf-model is specified, to determine which model file to download.
        Ignored if --model-type is jit or onnx with user-provided model files.
        """,
    )

    parser.add_argument(
        "--ctc",
        type=str2bool,
        default=False,
        help="Whether to use CTC head (instead of transducer).",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Decoding method, e.g., 'greedy_search'.",
    )

    # model path for JIT models and ONNX CTC models
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to the TorchScript model (for --model-type jit) or ONNX CTC model (for --model-type onnx, CTC).",
    )

    # ONNX transducer model paths
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        help="Path to the encoder ONNX model (for --model-type onnx, transducer).",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default="",
        help="Path to the decoder ONNX model (for --model-type onnx, transducer).",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        default="",
        help="Path to the joiner ONNX model (for --model-type onnx, transducer).",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Path to tokens.txt.",
    )

    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="HuggingFace repo ID, e.g., 'pkufool/zipformer-large'. "
        "If specified, the model and tokens will be downloaded from "
        "HuggingFace automatically.",
    )

    parser.add_argument(
        "--ms-model",
        type=str,
        default="",
        help="ModelScope repo ID, e.g., 'pkufool/zipformer-large'. "
        "If specified, the model and tokens will be downloaded from "
        "ModelScope automatically.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file.",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="*",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported.",
    )

    return parser


# ==============================================================================
# Audio I/O and feature extraction
# ==============================================================================


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list of 1-D float32 torch tensors."""
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        if sample_rate != expected_sample_rate:
            wave = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=expected_sample_rate
            )(wave)
        ans.append(wave[0].contiguous())
    return ans


def get_audio_durations(filenames: List[str]) -> List[float]:
    """Get duration in seconds for each audio file."""
    durations = []
    for f in filenames:
        info = torchaudio.info(f)
        durations.append(info.num_frames / info.sample_rate)
    return durations


# ==============================================================================
# Inference functions
# ==============================================================================


def _extract_features(sound_files: List[str], sample_rate: int, device: torch.device):
    """Extract Fbank features from input sound files.

    Always resamples to 16kHz for the model.
    """
    waves = read_sound_files(sound_files, expected_sample_rate=16000)
    waves = [w.to(device) for w in waves]

    features = []
    for w in waves:
        feat = torchaudio.compliance.kaldi.fbank(
            w.unsqueeze(0),
            num_mel_bins=80,
            sample_frequency=16000,
            dither=0,
            snip_edges=False,
            high_freq=-400,
        )  # (num_frames, 80)
        features.append(feat.to(device))
    feature_lengths = [f.size(0) for f in features]

    features = torch.nn.utils.rnn.pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )
    feature_lengths = torch.tensor(feature_lengths, device=device)
    return features, feature_lengths


def _infer_jit(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """JIT non-streaming transducer inference."""
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    durations = get_audio_durations(sound_files)

    start_time = time.time()

    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    hyps = greedy_search(model, encoder_out, encoder_out_lens)

    elapsed = time.time() - start_time

    results = []
    for filename, hyp, dur in zip(sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results, elapsed


def _infer_jit_streaming(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """JIT streaming transducer inference with dynamic batching.

    Uses DecodeStream to store intermediate encoder states and decoder output,
    and streaming_greedy_search for decoding. All audio features are extracted
    upfront (non-streaming), then processed chunk-by-chunk with dynamic batching.
    """
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)
    model.device = device

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    context_size = model.decoder.context_size
    chunk_length = model.encoder.chunk_size * 2
    pad_length = model.encoder.pad_length

    durations = get_audio_durations(sound_files)

    params = AttributeDict(
        {
            "blank_id": model.decoder.blank_id,
            "context_size": context_size,
            "decoding_method": decoding_method,
        }
    )

    num_streams = len(sound_files)
    all_streams = []
    decode_streams = []
    for idx in range(num_streams):
        initial_states = model.encoder.get_init_states(device=device)
        stream = DecodeStream(
            params=params,
            utt_id=sound_files[idx],
            initial_states=initial_states,
            device=device,
            pad_length=pad_length,
        )
        stream.set_features(features[idx, : feature_lengths[idx].item()])
        all_streams.append(stream)
        decode_streams.append(stream)

    tail_length = chunk_length + pad_length

    batch_states = stack_states([s.states for s in decode_streams])

    start_time = time.time()

    while len(decode_streams) > 0:
        batch_features = []
        batch_feature_lens = []
        for stream in decode_streams:
            feat, feat_len = stream.get_feature_frames(chunk_length)
            batch_features.append(feat)
            batch_feature_lens.append(feat_len)

        batch_feature_lens = torch.tensor(
            batch_feature_lens, dtype=torch.int32, device=device
        )
        batch_features = torch.nn.utils.rnn.pad_sequence(
            batch_features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )

        if batch_features.size(1) < tail_length:
            batch_features = torch.nn.functional.pad(
                batch_features,
                (0, 0, 0, tail_length - batch_features.size(1)),
                mode="constant",
                value=math.log(1e-10),
            )
        batch_feature_lens = torch.full_like(batch_feature_lens, tail_length)

        encoder_out, _, new_states = model.encoder(
            features=batch_features,
            feature_lengths=batch_feature_lens,
            states=batch_states,
        )

        encoder_out = model.joiner.encoder_proj(encoder_out)

        streaming_greedy_search(
            model=model,
            encoder_out=encoder_out,
            streams=decode_streams,
        )

        prev_count = len(decode_streams)
        done_mask = [s.done for s in decode_streams]
        decode_streams = [s for s, done in zip(decode_streams, done_mask) if not done]

        if len(decode_streams) < prev_count:
            if decode_streams:
                per_stream_states = unstack_states(new_states)
                remaining = [
                    st for st, done in zip(per_stream_states, done_mask) if not done
                ]
                batch_states = stack_states(remaining)
        else:
            batch_states = new_states

    elapsed = time.time() - start_time

    results = []
    for idx in range(num_streams):
        stream = all_streams[idx]
        hyp = stream.hyp[context_size:]
        text = token_ids_to_text(hyp, token_table) if hyp else ""
        results.append(
            {
                "filename": sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )
    return results, elapsed


def _infer_jit_ctc(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """JIT non-streaming CTC inference."""
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)

    token_table = SymbolTable.from_file(tokens)
    durations = get_audio_durations(sound_files)
    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    start_time = time.time()

    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    ctc_output = model.ctc_output(encoder_out)
    hyps = ctc_greedy_search(ctc_output, encoder_out_lens)

    results = []
    for filename, hyp, dur in zip(sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})

    elapsed = time.time() - start_time
    return results, elapsed


def _infer_jit_streaming_ctc(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """JIT streaming CTC inference with dynamic batching.

    Uses DecodeStream to store intermediate encoder states, and
    streaming_ctc_greedy_search for decoding. All audio features are extracted
    upfront (non-streaming), then processed chunk-by-chunk with dynamic batching.
    """
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)
    model.device = device

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    chunk_length = model.encoder.chunk_size * 2
    pad_length = model.encoder.pad_length

    durations = get_audio_durations(sound_files)

    params = AttributeDict(
        {
            "blank_id": 0,
            "context_size": 1,
            "decoding_method": decoding_method,
        }
    )

    num_streams = len(sound_files)
    all_streams = []
    decode_streams = []
    for idx in range(num_streams):
        initial_states = model.encoder.get_init_states(device=device)
        stream = DecodeStream(
            params=params,
            utt_id=sound_files[idx],
            initial_states=initial_states,
            device=device,
            pad_length=pad_length,
        )
        stream.set_features(features[idx, : feature_lengths[idx].item()])
        stream.hyp = []
        all_streams.append(stream)
        decode_streams.append(stream)

    tail_length = chunk_length + pad_length

    batch_states = stack_states([s.states for s in decode_streams])

    start_time = time.time()

    while len(decode_streams) > 0:
        batch_features = []
        batch_feature_lens = []
        for stream in decode_streams:
            feat, feat_len = stream.get_feature_frames(chunk_length)
            batch_features.append(feat)
            batch_feature_lens.append(feat_len)

        batch_feature_lens = torch.tensor(
            batch_feature_lens, dtype=torch.int32, device=device
        )
        batch_features = torch.nn.utils.rnn.pad_sequence(
            batch_features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )

        if batch_features.size(1) < tail_length:
            batch_features = torch.nn.functional.pad(
                batch_features,
                (0, 0, 0, tail_length - batch_features.size(1)),
                mode="constant",
                value=math.log(1e-10),
            )
        batch_feature_lens = torch.full_like(batch_feature_lens, tail_length)

        encoder_out, encoder_out_lens, new_states = model.encoder(
            features=batch_features,
            feature_lengths=batch_feature_lens,
            states=batch_states,
        )

        streaming_ctc_greedy_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            streams=decode_streams,
        )

        prev_count = len(decode_streams)
        done_mask = [s.done for s in decode_streams]
        decode_streams = [s for s, done in zip(decode_streams, done_mask) if not done]

        if len(decode_streams) < prev_count:
            if decode_streams:
                per_stream_states = unstack_states(new_states)
                remaining = [
                    st for st, done in zip(per_stream_states, done_mask) if not done
                ]
                batch_states = stack_states(remaining)
        else:
            batch_states = new_states

    elapsed = time.time() - start_time

    results = []
    for idx in range(num_streams):
        stream = all_streams[idx]
        text = token_ids_to_text(stream.hyp, token_table) if stream.hyp else ""
        results.append(
            {
                "filename": sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )
    return results, elapsed


def _infer_onnx(
    encoder: str,
    decoder: str,
    joiner: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """ONNX non-streaming transducer inference."""

    model = OnnxTransducerModel(encoder, decoder, joiner)
    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    durations = get_audio_durations(sound_files)

    start_time = time.time()

    encoder_out, encoder_out_lens = model.run_encoder(features, feature_lengths)

    hyps = greedy_search(model, encoder_out, encoder_out_lens)

    elapsed = time.time() - start_time

    results = []
    for filename, hyp, dur in zip(sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results, elapsed


def _infer_onnx_ctc(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """ONNX non-streaming CTC inference."""
    model = OnnxCtcModel(model_path)

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)
    features = features.cpu()
    feature_lengths = feature_lengths.cpu()

    token_table = SymbolTable.from_file(tokens)
    durations = get_audio_durations(sound_files)

    start_time = time.time()

    log_probs, log_probs_len = model(features, feature_lengths)

    hyps = ctc_greedy_search(log_probs, log_probs_len)

    elapsed = time.time() - start_time

    results = []
    for filename, hyp, dur in zip(sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results, elapsed


def _infer_onnx_streaming(
    encoder: str,
    decoder: str,
    joiner: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """ONNX streaming transducer inference with DecodeStream.

    Processes each utterance sequentially (ONNX streaming models are batch_size=1).
    Uses DecodeStream to manage features and streaming_greedy_search for decoding.
    Encoder states are stored in stream.states and restored before each chunk.
    """
    model = OnnxStreamingTransducerModel(encoder, decoder, joiner)

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    context_size = model.context_size
    segment = model.segment
    offset = model.offset
    durations = get_audio_durations(sound_files)

    params = AttributeDict(
        {
            "blank_id": getattr(model, "blank_id", 0),
            "context_size": context_size,
            "decoding_method": decoding_method,
        }
    )

    start_time = time.time()

    results = []
    for idx in range(len(sound_files)):
        model.reset_states()
        stream = DecodeStream(
            params=params,
            utt_id=sound_files[idx],
            initial_states=list(model.states),
            device=torch.device("cpu"),
            pad_length=segment - offset,
        )
        feat = features[idx, : feature_lengths[idx].item()].cpu()
        stream.set_features(feat)

        while not stream.done:
            chunk_feat, chunk_len = stream.get_feature_frames(offset)
            if chunk_len < segment:
                chunk_feat = torch.nn.functional.pad(
                    chunk_feat,
                    (0, 0, 0, segment - chunk_len),
                    mode="constant",
                    value=math.log(1e-10),
                )
            chunk_feat = chunk_feat.unsqueeze(0)  # (1, segment, feat_dim)
            encoder_out = model.run_encoder(chunk_feat)  # (1, T', C)
            if encoder_out.ndim == 2:
                encoder_out = encoder_out.unsqueeze(0)

            streaming_greedy_search(
                model=model,
                encoder_out=encoder_out,
                streams=[stream],
            )

        hyp = stream.hyp[context_size:]
        text = token_ids_to_text(hyp, token_table) if hyp else ""
        results.append(
            {
                "filename": sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )

    elapsed = time.time() - start_time
    return results, elapsed


def _infer_onnx_streaming_ctc(
    model_path: str,
    tokens: str,
    sound_files: List[str],
    sample_rate: int,
    device: torch.device,
    decoding_method: str,
) -> List[dict]:
    """ONNX streaming CTC inference with DecodeStream.

    Processes each utterance sequentially (ONNX streaming models are batch_size=1).
    Uses DecodeStream to manage features and streaming_ctc_greedy_search for decoding.
    Encoder states are stored in stream.states and restored before each chunk.
    """
    model = OnnxStreamingCtcModel(model_path)

    features, feature_lengths = _extract_features(sound_files, sample_rate, device)

    token_table = SymbolTable.from_file(tokens)
    segment = model.segment
    offset = model.offset
    durations = get_audio_durations(sound_files)

    params = AttributeDict(
        {
            "blank_id": 0,
            "context_size": 1,
            "decoding_method": decoding_method,
        }
    )

    start_time = time.time()

    results = []
    for idx in range(len(sound_files)):
        model.reset_states()

        stream = DecodeStream(
            params=params,
            utt_id=sound_files[idx],
            initial_states=list(model.states),
            device=torch.device("cpu"),
            pad_length=segment - offset,
        )
        feat = features[idx, : feature_lengths[idx].item()].cpu()
        stream.set_features(feat)
        stream.hyp = []

        while not stream.done:
            chunk_feat, chunk_len = stream.get_feature_frames(offset)
            if chunk_len < segment:
                chunk_feat = torch.nn.functional.pad(
                    chunk_feat,
                    (0, 0, 0, segment - chunk_len),
                    mode="constant",
                    value=math.log(1e-10),
                )
            chunk_feat = chunk_feat.unsqueeze(0)  # (1, segment, feat_dim)

            ctc_log_probs = model(chunk_feat)  # (1, T', vocab_size)

            if ctc_log_probs.ndim == 2:
                ctc_log_probs = ctc_log_probs.unsqueeze(0)
            T = ctc_log_probs.size(1)
            encoder_out_lens = torch.tensor([T], dtype=torch.int32)

            streaming_ctc_greedy_search(
                model=model,
                encoder_out=ctc_log_probs,
                encoder_out_lens=encoder_out_lens,
                streams=[stream],
            )

        text = token_ids_to_text(stream.hyp, token_table) if stream.hyp else ""
        results.append(
            {
                "filename": sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )

    elapsed = time.time() - start_time
    return results, elapsed


# ==============================================================================
# Output
# ==============================================================================


def print_results(results: List[dict], elapsed: float):
    """Print decoding results and RTF."""
    total_duration = 0.0
    for r in results:
        total_duration += r["duration"]
        logging.info(f"{r['filename']} ({r['duration']:.2f}s):\n  {r['text']}\n")

    rtf = elapsed / total_duration if total_duration > 0 else float("inf")

    logging.info(
        f"Processed {len(results)} file(s) (total audio: {total_duration:.2f}s)\n"
        f"Processing time: {elapsed:.2f}s\n"
        f"RTF: {rtf:.4f}"
    )


# ==============================================================================
# Model download
# ==============================================================================


def _get_model_filenames(
    model_type: str,
    dtype: str,
    streaming: bool,
    ctc: bool,
    chunk_size: int,
    left_context_frames: int,
) -> dict:
    """Resolve file names to download based on model configuration.

    Returns a dict mapping path keys (tokens, model, encoder, decoder, joiner)
    to their relative file paths in the repo.
    """
    filenames = {"tokens": "data/tokens.txt"}

    if model_type == "jit":
        if streaming:
            filenames["model"] = (
                f"jit_model-chunk-{chunk_size}-left-{left_context_frames}.pt"
            )
        else:
            filenames["model"] = "jit_model.pt"
    elif model_type == "onnx":
        if ctc:
            if streaming:
                filename = f"ctc-chunk-{chunk_size}-left-{left_context_frames}"
                if dtype == "fp16":
                    filenames["model"] = filename + ".fp16.onnx"
                elif dtype == "int8":
                    filenames["model"] = filename + ".int8.onnx"
                else:
                    filenames["model"] = filename + ".onnx"
            else:
                if dtype == "fp16":
                    filenames["model"] = "ctc.fp16.onnx"
                elif dtype == "int8":
                    filenames["model"] = "ctc.int8.onnx"
                else:
                    filenames["model"] = "ctc.onnx"
        else:
            if streaming:
                encoder_filename = (
                    f"encoder-chunk-{chunk_size}-left-{left_context_frames}"
                )
                decoder_filename = (
                    f"decoder-chunk-{chunk_size}-left-{left_context_frames}"
                )
                joiner_filename = (
                    f"joiner-chunk-{chunk_size}-left-{left_context_frames}"
                )
                if dtype == "fp16":
                    filenames["encoder"] = encoder_filename + ".fp16.onnx"
                    filenames["decoder"] = decoder_filename + ".onnx"
                    filenames["joiner"] = joiner_filename + ".fp16.onnx"
                elif dtype == "int8":
                    filenames["encoder"] = encoder_filename + ".int8.onnx"
                    filenames["decoder"] = decoder_filename + ".onnx"
                    filenames["joiner"] = joiner_filename + ".int8.onnx"
                else:
                    filenames["encoder"] = encoder_filename + ".onnx"
                    filenames["decoder"] = decoder_filename + ".onnx"
                    filenames["joiner"] = joiner_filename + ".onnx"
            else:
                if dtype == "fp16":
                    filenames["encoder"] = "encoder.fp16.onnx"
                    filenames["decoder"] = "decoder.onnx"
                    filenames["joiner"] = "joiner.fp16.onnx"
                elif dtype == "int8":
                    filenames["encoder"] = "encoder.int8.onnx"
                    filenames["decoder"] = "decoder.onnx"
                    filenames["joiner"] = "joiner.int8.onnx"
                else:
                    filenames["encoder"] = "encoder.onnx"
                    filenames["decoder"] = "decoder.onnx"
                    filenames["joiner"] = "joiner.onnx"
    return filenames


def _download_model(
    hf_model: str,
    ms_model: str,
    model_type: str,
    dtype: str,
    streaming: bool,
    ctc: bool,
    chunk_size: int,
    left_context_frames: int,
) -> dict:
    """Download model files from HuggingFace or ModelScope.

    Returns a dict with resolved local paths: {tokens, model, encoder, decoder, joiner}.
    """
    filenames = _get_model_filenames(
        model_type, dtype, streaming, ctc, chunk_size, left_context_frames
    )

    paths = {}
    if hf_model:
        from huggingface_hub import hf_hub_download

        for key, filename in filenames.items():
            paths[key] = hf_hub_download(repo_id=hf_model, filename=filename)
        logging.info(f"Downloaded HuggingFace model '{hf_model}': {paths}")
    elif ms_model:
        from modelscope.hub.file_download import model_file_download

        for key, filename in filenames.items():
            paths[key] = model_file_download(model_id=ms_model, file_path=filename)
        logging.info(f"Downloaded ModelScope model '{ms_model}': {paths}")

    return paths


# ==============================================================================
# Unified Python API
# ==============================================================================


@torch.no_grad()
def inference(
    sound_files: List[str],
    *,
    model_type: str = "jit",
    streaming: bool = False,
    ctc: bool = False,
    decoding_method: str = "greedy_search",
    model: str = "",
    encoder: str = "",
    decoder: str = "",
    joiner: str = "",
    tokens: str = "",
    hf_model: str = "",
    ms_model: str = "",
    dtype: str = "fp32",
    chunk_size: int = 32,
    left_context_frames: int = 128,
    sample_rate: int = 16000,
    device: str = "",
) -> List[dict]:
    """Unified inference API for Zipformer models.

    Args:
      sound_files: List of audio file paths to transcribe.
      model_type: Model format, "jit" or "onnx".
      streaming: Whether to use streaming inference.
      ctc: Whether to use CTC head (instead of transducer).
      decoding_method: Decoding method, e.g. "greedy_search".
      model: Path to JIT model or ONNX CTC model.
      encoder: Path to ONNX transducer encoder.
      decoder: Path to ONNX transducer decoder.
      joiner: Path to ONNX transducer joiner.
      tokens: Path to tokens.txt.
      hf_model: HuggingFace repo ID. If set, overrides local model paths.
      ms_model: ModelScope model ID. If set, overrides local model paths.
      dtype: Data type for hub download (fp32/fp16/int8).
      chunk_size: Chunk size for streaming (used for hub download file selection).
      left_context_frames: Left context frames for streaming (used for hub download).
      sample_rate: Expected sample rate of input audio (resampled to 16kHz internally).
      device: Device string ("cuda", "cpu", or "" for auto).

    Returns:
      A tuple (results, elapsed) where results is a list of dicts with keys:
      filename, text, duration; and elapsed is the inference + decoding time in seconds.
    """
    # Resolve device
    if device:
        device = torch.device(device)
    else:
        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # hf_model / ms_model have higher priority — override local paths
    if hf_model or ms_model:
        paths = _download_model(
            hf_model,
            ms_model,
            model_type,
            dtype,
            streaming,
            ctc,
            chunk_size,
            left_context_frames,
        )
        tokens = paths["tokens"]
        model = paths.get("model", "")
        encoder = paths.get("encoder", "")
        decoder = paths.get("decoder", "")
        joiner = paths.get("joiner", "")

    if not tokens:
        raise ValueError(
            "tokens is required. Provide --tokens or use --hf-model/--ms-model."
        )

    # ONNX models run on CPU
    if model_type == "onnx":
        device = torch.device("cpu")

    # Dispatch to the appropriate internal function
    if model_type == "jit":
        if streaming and ctc:
            return _infer_jit_streaming_ctc(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        elif streaming:
            return _infer_jit_streaming(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        elif ctc:
            return _infer_jit_ctc(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        else:
            return _infer_jit(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
    elif model_type == "onnx":
        if streaming and ctc:
            return _infer_onnx_streaming_ctc(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        elif streaming:
            return _infer_onnx_streaming(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        elif ctc:
            return _infer_onnx_ctc(
                model_path=model,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
        else:
            return _infer_onnx(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                sound_files=sound_files,
                sample_rate=sample_rate,
                device=device,
                decoding_method=decoding_method,
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ==============================================================================
# Main (CLI entry point)
# ==============================================================================


@torch.no_grad()
def main():
    args = get_parser().parse_args()

    logging.info(vars(args))

    results, elapsed = inference(
        sound_files=args.sound_files,
        model_type=args.model_type,
        streaming=args.streaming,
        ctc=args.ctc,
        decoding_method=args.decoding_method,
        model=args.model,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        hf_model=args.hf_model,
        ms_model=args.ms_model,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        left_context_frames=args.left_context_frames,
        sample_rate=args.sample_rate,
    )
    print_results(results, elapsed)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
