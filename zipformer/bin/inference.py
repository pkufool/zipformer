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

  python inference.py \\
    --model-type jit \\
    --nn-model-filename ./exp/jit_script.pt \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(2) JIT streaming transducer:

  python inference.py \\
    --model-type jit --streaming true \\
    --nn-model-filename ./exp/jit_script_chunk_16_left_128.pt \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(3) ONNX non-streaming transducer:

  python inference.py \\
    --model-type onnx \\
    --encoder-model-filename ./exp/encoder.onnx \\
    --decoder-model-filename ./exp/decoder.onnx \\
    --joiner-model-filename ./exp/joiner.onnx \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(4) ONNX non-streaming CTC:

  python inference.py \\
    --model-type onnx --ctc true \\
    --nn-model ./exp/model.onnx \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(5) ONNX streaming transducer:

  python inference.py \\
    --model-type onnx --streaming true \\
    --encoder-model-filename ./exp/encoder-streaming.onnx \\
    --decoder-model-filename ./exp/decoder-streaming.onnx \\
    --joiner-model-filename ./exp/joiner-streaming.onnx \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(6) ONNX streaming CTC:

  python inference.py \\
    --model-type onnx --streaming true --ctc true \\
    --nn-model ./exp/ctc-streaming.onnx \\
    --tokens ./data/lang_bpe_500/tokens.txt \\
    /path/to/foo.wav /path/to/bar.wav

(7) Download model from HuggingFace (JIT):

  python inference.py \\
    --model-type jit \\
    --hf-model ks-fsa/zipformer-medium-v1 \\
    /path/to/foo.wav /path/to/bar.wav

(8) Download model from HuggingFace (ONNX transducer):

  python inference.py \\
    --model-type onnx \\
    --hf-model ks-fsa/zipformer-medium-v1 \\
    /path/to/foo.wav /path/to/bar.wav
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import kaldi_native_fbank as knf

from zipformer.utils import str2bool, SymbolTable, AttributeDict
from zipformer.decode.search import greedy_search, streaming_greedy_search
from zipformer.decode.stream import DecodeStream

# ==============================================================================
# Argument parsing
# ==============================================================================


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["checkpoint", "jit", "onnx"],
        default="checkpoint",
        help="Model format: 'checkpoint' for PyTorch checkpoint, 'jit' for TorchScript, 'onnx' for ONNX.",
    )

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=False,
        help="Whether to use streaming inference.",
    )

    parser.add_argument(
        "--ctc",
        type=str2bool,
        default=False,
        help="Whether to use CTC head (instead of transducer).",
    )

    # JIT model path
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to the TorchScript model (for --model-type jit).",
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
        help="Path to tokens.txt. If not provided and --hf-model is set, "
        "defaults to data/lang_bpe_500/tokens.txt inside the repo.",
    )

    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="HuggingFace repo ID, e.g., 'ks-fsa/zipformer-medium-v1'. "
        "If specified, the model and tokens will be downloaded from "
        "HuggingFace automatically.",
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
# HuggingFace model download
# ==============================================================================


def download_hf_model(repo_id: str) -> Path:
    """Download a HuggingFace model repo using huggingface_hub.

    Returns the local cache directory path where the repo is downloaded.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id)
    logging.info(f"Downloaded HuggingFace model '{repo_id}' to {local_dir}")
    return Path(local_dir)


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
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        )
        ans.append(wave[0].contiguous())
    return ans


def compute_fbank(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Compute fbank features for a single waveform.

    Args:
      waveform:
        A 1-D float32 tensor of audio samples.
      sample_rate:
        The sample rate of the audio.
    Returns:
      Return a 2-D tensor of shape (num_frames, feature_dim).
    """
    feat = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=sample_rate,
        dither=0,
        snip_edges=False,
        high_freq=-400,
    )
    return feat


def get_audio_durations(filenames: List[str]) -> List[float]:
    """Get duration in seconds for each audio file."""
    durations = []
    for f in filenames:
        info = torchaudio.info(f)
        durations.append(info.num_frames / info.sample_rate)
    return durations


# ==============================================================================
# Token decoding
# ==============================================================================

def token_ids_to_text(token_ids: List[int], token_table) -> str:
    """Convert token IDs to text using a k2 SymbolTable."""
    text = ""
    for i in token_ids:
        text += token_table[i]
    return text.replace("▁", " ").strip()


def token_ids_to_text_bpe(token_ids: List[int], tokens_path: str) -> str:
    """Convert token IDs to text handling byte-level BPE."""
    id2token = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            token, idx = line.split()
            if token[:3] == "<0x" and token[-1] == ">":
                token = int(token[1:-1], base=16)
                assert 0 <= token < 256, token
                token = token.to_bytes(1, byteorder="little")
            else:
                token = token.encode(encoding="utf-8")
            id2token[int(idx)] = token

    text = b""
    for i in token_ids:
        text += id2token[i]
    return text.decode(encoding="utf-8").replace("▁", " ").strip()


# ==============================================================================
# Inference functions
# ==============================================================================

def extract_features(args):
    """Extract Fbank features from input sound files."""
    waves = read_sound_files(args.sound_files, args.sample_rate)
    waves = [w.to(args.device) for w in waves]

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
        features.append(feat.to(args.device))
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )
    feature_lengths = torch.tensor(feature_lengths, device=args.device)
    return features, feature_lengths


def infer_jit(args) -> List[dict]:
    """JIT non-streaming transducer inference."""
    model = torch.jit.load(args.model)
    model.eval()
    model.to(args.device)

    features, feature_lengths = extract_features(args)

    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    hyps = greedy_search(model, encoder_out, encoder_out_lens)

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def stack_states(state_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack list of zipformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zipformer when those utterances are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance. For element-n,
        state_list[n] is a list of cached tensors of all encoder layers. For layer-i,
        state_list[n][i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1,
        cached_val2, cached_conv1, cached_conv2).
        state_list[n][-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
        state_list[n][-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Note:
      It is the inverse of :func:`unstack_states`.
    """
    batch_size = len(state_list)
    assert (len(state_list[0]) - 2) % 6 == 0, len(state_list[0])
    tot_num_layers = (len(state_list[0]) - 2) // 6

    batch_states = []
    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key = torch.cat(
            [state_list[i][layer_offset] for i in range(batch_size)], dim=1
        )
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn = torch.cat(
            [state_list[i][layer_offset + 1] for i in range(batch_size)], dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1 = torch.cat(
            [state_list[i][layer_offset + 2] for i in range(batch_size)], dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2 = torch.cat(
            [state_list[i][layer_offset + 3] for i in range(batch_size)], dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1 = torch.cat(
            [state_list[i][layer_offset + 4] for i in range(batch_size)], dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2 = torch.cat(
            [state_list[i][layer_offset + 5] for i in range(batch_size)], dim=0
        )
        batch_states += [
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ]

    cached_embed_left_pad = torch.cat(
        [state_list[i][-2] for i in range(batch_size)], dim=0
    )
    batch_states.append(cached_embed_left_pad)

    processed_lens = torch.cat([state_list[i][-1] for i in range(batch_size)], dim=0)
    batch_states.append(processed_lens)

    return batch_states


def unstack_states(batch_states: List[torch.Tensor]) -> List[List[torch.Tensor]]:
    """Unstack the zipformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Args:
        batch_states: A list of cached tensors of all encoder layers. For layer-i,
          states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
          cached_conv1, cached_conv2).
          state_list[-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
          states[-1] is processed_lens of shape (batch,), which records the number
          of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Returns:
        state_list: A list of list. Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance.
    """
    assert (len(batch_states) - 2) % 6 == 0, len(batch_states)
    tot_num_layers = (len(batch_states) - 2) // 6

    processed_lens = batch_states[-1]
    batch_size = processed_lens.shape[0]

    state_list = [[] for _ in range(batch_size)]

    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key_list = batch_states[layer_offset].chunk(chunks=batch_size, dim=1)
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn_list = batch_states[layer_offset + 1].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1_list = batch_states[layer_offset + 2].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2_list = batch_states[layer_offset + 3].chunk(
            chunks=batch_size, dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1_list = batch_states[layer_offset + 4].chunk(
            chunks=batch_size, dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2_list = batch_states[layer_offset + 5].chunk(
            chunks=batch_size, dim=0
        )
        for i in range(batch_size):
            state_list[i] += [
                cached_key_list[i],
                cached_nonlin_attn_list[i],
                cached_val1_list[i],
                cached_val2_list[i],
                cached_conv1_list[i],
                cached_conv2_list[i],
            ]

    cached_embed_left_pad_list = batch_states[-2].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(cached_embed_left_pad_list[i])

    processed_lens_list = batch_states[-1].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(processed_lens_list[i])

    return state_list


def infer_jit_streaming(args) -> List[dict]:
    """JIT streaming transducer inference with dynamic batching.

    Uses DecodeStream to store intermediate encoder states and decoder output,
    and streaming_greedy_search for decoding. All audio features are extracted
    upfront (non-streaming), then processed chunk-by-chunk with dynamic batching.
    """
    model = torch.jit.load(args.model)
    model.eval()
    model.to(args.device)
    model.device = args.device

    features, feature_lengths = extract_features(args)

    token_table = SymbolTable.from_file(args.tokens)
    context_size = model.decoder.context_size
    chunk_length = model.encoder.chunk_size * 2
    pad_length = model.encoder.pad_length

    durations = get_audio_durations(args.sound_files)

    params = AttributeDict(
        {
            "blank_id": model.decoder.blank_id,
            "context_size": context_size,
            "decoding_method": "greedy_search",
        }
    )

    num_streams = len(args.sound_files)
    all_streams = []
    decode_streams = []
    for idx in range(num_streams):
        initial_states = model.encoder.get_init_states(device=args.device)
        stream = DecodeStream(
            params=params,
            utt_id=args.sound_files[idx],
            initial_states=initial_states,
            device=args.device,
            pad_length=pad_length,
        )
        stream.set_features(features[idx, : feature_lengths[idx].item()])
        all_streams.append(stream)
        decode_streams.append(stream)

    tail_length = chunk_length + pad_length

    while len(decode_streams) > 0:
        batch_features = []
        batch_feature_lens = []
        for stream in decode_streams:
            feat, feat_len = stream.get_feature_frames(chunk_length)
            batch_features.append(feat)
            batch_feature_lens.append(feat_len)

        batch_feature_lens = torch.tensor(
            batch_feature_lens, dtype=torch.int32, device=args.device
        )
        batch_features = pad_sequence(
            batch_features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )

        # Pad features to tail_length and set all feature_lens to tail_length
        # so encoder always produces chunk_size output frames for every stream.
        if batch_features.size(1) < tail_length:
            batch_features = torch.nn.functional.pad(
                batch_features,
                (0, 0, 0, tail_length - batch_features.size(1)),
                mode="constant",
                value=math.log(1e-10),
            )
        batch_feature_lens = torch.full_like(
            batch_feature_lens, tail_length
        )

        stacked_states = stack_states([s.states for s in decode_streams])

        encoder_out, encoder_out_lens, new_states = model.encoder(
            features=batch_features,
            feature_lengths=batch_feature_lens,
            states=stacked_states,
        )

        per_stream_states = unstack_states(new_states)
        for j, stream in enumerate(decode_streams):
            stream.states = per_stream_states[j]

        encoder_out = model.joiner.encoder_proj(encoder_out)

        streaming_greedy_search(
            model=model,
            encoder_out=encoder_out,
            streams=decode_streams,
        )

        decode_streams = [s for s in decode_streams if not s.done]

    results = []
    for idx in range(num_streams):
        stream = all_streams[idx]
        hyp = stream.hyp[context_size:]
        text = token_ids_to_text(hyp, token_table) if hyp else ""
        results.append(
            {"filename": args.sound_files[idx], "text": text, "duration": durations[idx]}
        )

    return results


def infer_onnx(args) -> List[dict]:
    """ONNX non-streaming transducer inference."""
    from torch.nn.utils.rnn import pad_sequence

    model = OnnxTransducerModel(
        args.encoder_model_filename,
        args.decoder_model_filename,
        args.joiner_model_filename,
    )

    fbank = create_fbank(args.sample_rate)
    waves = read_sound_files(args.sound_files, args.sample_rate)

    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)

    encoder_out, encoder_out_lens = model.run_encoder(features, feature_lengths)
    hyps = greedy_search_transducer_batch(model, encoder_out, encoder_out_lens)

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files, args.sample_rate)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def infer_onnx_ctc(args) -> List[dict]:
    """ONNX non-streaming CTC inference."""
    from torch.nn.utils.rnn import pad_sequence

    model = OnnxCtcModel(args.nn_model)

    fbank = create_fbank(args.sample_rate)
    waves = read_sound_files(args.sound_files, args.sample_rate)

    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)

    log_probs, log_probs_len = model(features, feature_lengths)

    token_table = load_token_table(args.tokens)
    durations = get_audio_durations(args.sound_files, args.sample_rate)
    blank_id = 0

    results = []
    for i in range(log_probs.size(0)):
        indexes = log_probs[i, : log_probs_len[i]].argmax(dim=-1)
        token_ids = torch.unique_consecutive(indexes)
        token_ids = token_ids[token_ids != blank_id].tolist()
        text = token_ids_to_text(token_ids, token_table)
        results.append(
            {
                "filename": args.sound_files[i],
                "text": text,
                "duration": durations[i],
            }
        )
    return results


def infer_onnx_streaming(args) -> List[dict]:
    """ONNX streaming transducer inference."""
    model = OnnxStreamingTransducerModel(
        args.encoder_model_filename,
        args.decoder_model_filename,
        args.joiner_model_filename,
    )

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files, args.sample_rate)
    sample_rate = args.sample_rate
    results = []

    for idx, sound_file in enumerate(args.sound_files):
        model.reset_states()
        online_fbank = create_streaming_fbank(sample_rate)
        wave_samples = read_sound_files([sound_file], sample_rate)[0]

        tail_padding = torch.zeros(int(0.3 * sample_rate), dtype=torch.float32)
        wave_samples = torch.cat([wave_samples, tail_padding])

        num_processed_frames = 0
        segment = model.segment
        offset = model.offset
        context_size = model.context_size
        hyp = None
        decoder_out = None

        chunk = int(1 * sample_rate)
        start = 0
        while start < wave_samples.numel():
            end = min(start + chunk, wave_samples.numel())
            samples = wave_samples[start:end]
            start += chunk

            online_fbank.accept_waveform(sampling_rate=sample_rate, waveform=samples)

            while online_fbank.num_frames_ready - num_processed_frames >= segment:
                frames = []
                for i in range(segment):
                    frames.append(online_fbank.get_frame(num_processed_frames + i))
                num_processed_frames += offset
                frames = torch.cat(frames, dim=0).unsqueeze(0)
                encoder_out = model.run_encoder(frames)
                hyp, decoder_out = greedy_search_transducer_streaming_onnx(
                    model,
                    encoder_out,
                    context_size,
                    decoder_out,
                    hyp,
                )

        if hyp is not None:
            text = token_ids_to_text(hyp[context_size:], token_table)
        else:
            text = ""
        results.append(
            {"filename": sound_file, "text": text, "duration": durations[idx]}
        )

    return results


def infer_onnx_streaming_ctc(args) -> List[dict]:
    """ONNX streaming CTC inference."""
    model = OnnxStreamingCtcModel(args.nn_model)

    durations = get_audio_durations(args.sound_files, args.sample_rate)
    sample_rate = args.sample_rate
    results = []

    for idx, sound_file in enumerate(args.sound_files):
        model.reset_states()
        online_fbank = create_streaming_fbank(sample_rate)
        wave_samples = read_sound_files([sound_file], sample_rate)[0]

        tail_padding = torch.zeros(int(0.3 * sample_rate), dtype=torch.float32)
        wave_samples = torch.cat([wave_samples, tail_padding])

        num_processed_frames = 0
        segment = model.segment
        offset = model.offset
        hyp = []

        chunk = int(1 * sample_rate)
        start = 0
        while start < wave_samples.numel():
            end = min(start + chunk, wave_samples.numel())
            samples = wave_samples[start:end]
            start += chunk

            online_fbank.accept_waveform(sampling_rate=sample_rate, waveform=samples)

            while online_fbank.num_frames_ready - num_processed_frames >= segment:
                frames = []
                for i in range(segment):
                    frames.append(online_fbank.get_frame(num_processed_frames + i))
                num_processed_frames += offset
                frames = torch.cat(frames, dim=0).unsqueeze(0)
                log_probs = model(frames)
                hyp += greedy_search_ctc(log_probs)

        text = token_ids_to_text_bpe(hyp, args.tokens)
        results.append(
            {"filename": sound_file, "text": text, "duration": durations[idx]}
        )

    return results


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
# Main
# ==============================================================================


def _resolve_hf_model_paths(args, hf_dir: Path):
    """When --hf-model is used, auto-resolve model file paths from the repo."""
    import glob

    # Auto-resolve tokens
    if not args.tokens:
        tokens_path = hf_dir / "data" / "lang_bpe_500" / "tokens.txt"
        if tokens_path.exists():
            args.tokens = str(tokens_path)
            logging.info(f"Auto-resolved tokens: {args.tokens}")
        else:
            raise ValueError(
                f"--tokens not provided and default path not found in repo: "
                f"{tokens_path}. Please specify --tokens explicitly."
            )

    # Auto-resolve model file paths
    if args.model_type == "jit":
        if not args.nn_model_filename:
            # Try common JIT model naming patterns
            candidates = [
                "exp/jit_script.pt",
                "exp/jit_script_chunk_16_left_128.pt",
            ]
            for cand in candidates:
                p = hf_dir / cand
                if p.exists():
                    args.nn_model_filename = str(p)
                    logging.info(f"Auto-resolved JIT model: {args.nn_model_filename}")
                    break
            if not args.nn_model_filename:
                raise ValueError(
                    f"--nn-model-filename not provided and no JIT model found in "
                    f"repo. Please specify --nn-model-filename explicitly."
                )
    elif args.model_type == "onnx":
        if args.ctc:
            if not args.nn_model:
                candidates = ["exp/model.onnx", "exp/ctc.onnx"]
                for cand in candidates:
                    p = hf_dir / cand
                    if p.exists():
                        args.nn_model = str(p)
                        logging.info(f"Auto-resolved ONNX CTC model: {args.nn_model}")
                        break
                if not args.nn_model:
                    raise ValueError(
                        f"--nn-model not provided and no ONNX CTC model found in "
                        f"repo. Please specify --nn-model explicitly."
                    )
        else:
            # Transducer: resolve encoder, decoder, joiner
            if not args.encoder_model_filename:
                for pattern in ["exp/encoder*.onnx", "exp/encoder-*.onnx"]:
                    matches = sorted(glob.glob(str(hf_dir / pattern)))
                    if matches:
                        args.encoder_model_filename = matches[-1]
                        logging.info(
                            f"Auto-resolved encoder: {args.encoder_model_filename}"
                        )
                        break
            if not args.decoder_model_filename:
                for pattern in ["exp/decoder*.onnx", "exp/decoder-*.onnx"]:
                    matches = sorted(glob.glob(str(hf_dir / pattern)))
                    if matches:
                        args.decoder_model_filename = matches[-1]
                        logging.info(
                            f"Auto-resolved decoder: {args.decoder_model_filename}"
                        )
                        break
            if not args.joiner_model_filename:
                for pattern in ["exp/joiner*.onnx", "exp/joiner-*.onnx"]:
                    matches = sorted(glob.glob(str(hf_dir / pattern)))
                    if matches:
                        args.joiner_model_filename = matches[-1]
                        logging.info(
                            f"Auto-resolved joiner: {args.joiner_model_filename}"
                        )
                        break

            missing = []
            if not args.encoder_model_filename:
                missing.append("--encoder-model-filename")
            if not args.decoder_model_filename:
                missing.append("--decoder-model-filename")
            if not args.joiner_model_filename:
                missing.append("--joiner-model-filename")
            if missing:
                raise ValueError(
                    f"Could not auto-resolve: {', '.join(missing)}. "
                    f"Please specify them explicitly."
                )


@torch.no_grad()
def main():
    args = get_parser().parse_args()

    if args.hf_model:
        hf_dir = download_hf_model(args.hf_model)
        _resolve_hf_model_paths(args, hf_dir)

    if not args.tokens:
        raise ValueError("--tokens is required when --hf-model is not used.")

        
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    args.device = device

    logging.info(vars(args))

    start_time = time.time()

    if args.model_type == "jit":
        if args.streaming:
            results = infer_jit_streaming(args)
        else:
            results = infer_jit(args)
    elif args.model_type == "onnx":
        if args.streaming:
            if args.ctc:
                results = infer_onnx_streaming_ctc(args)
            else:
                results = infer_onnx_streaming(args)
        elif args.ctc:
            results = infer_onnx_ctc(args)
        else:
            results = infer_onnx(args)

    elapsed = time.time() - start_time
    print_results(results, elapsed)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
