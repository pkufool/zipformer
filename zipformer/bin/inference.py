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

from zipformer.utils import str2bool, SymbolTable, AttributeDict, token_ids_to_text
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
        """
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
            wave = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)(wave)
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

    features = torch.nn.utils.rnn.pad_sequence(
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

    # Pre-stack initial states; maintain batched states across iterations
    # to avoid redundant stack/unstack when batch composition is unchanged.
    batch_states = stack_states([s.states for s in decode_streams])

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
        batch_features = torch.nn.utils.rnn.pad_sequence(
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
            # Batch composition changed - unstack and re-stack only remaining streams
            if decode_streams:
                per_stream_states = unstack_states(new_states)
                remaining = [
                    st for st, done in zip(per_stream_states, done_mask) if not done
                ]
                batch_states = stack_states(remaining)
        else:
            # Batch unchanged - reuse encoder output states directly
            batch_states = new_states

    results = []
    for idx in range(num_streams):
        stream = all_streams[idx]
        hyp = stream.hyp[context_size:]
        text = token_ids_to_text(hyp, token_table) if hyp else ""
        results.append(
            {
                "filename": args.sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )
    return results


def infer_jit_ctc(args) -> List[dict]:
    """JIT non-streaming CTC inference."""
    model = torch.jit.load(args.model)
    model.eval()
    model.to(args.device)

    features, feature_lengths = extract_features(args)

    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    ctc_output = model.ctc_output(encoder_out)
    hyps = ctc_greedy_search(ctc_output, encoder_out_lens)

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def infer_jit_streaming_ctc(args) -> List[dict]:
    """JIT streaming CTC inference with dynamic batching.

    Uses DecodeStream to store intermediate encoder states, and
    streaming_ctc_greedy_search for decoding. All audio features are extracted
    upfront (non-streaming), then processed chunk-by-chunk with dynamic batching.
    """
    model = torch.jit.load(args.model)
    model.eval()
    model.to(args.device)
    model.device = args.device

    features, feature_lengths = extract_features(args)

    token_table = SymbolTable.from_file(args.tokens)
    chunk_length = model.encoder.chunk_size * 2
    pad_length = model.encoder.pad_length

    durations = get_audio_durations(args.sound_files)

    params = AttributeDict(
        {
            "blank_id": 0,
            "context_size": 1,
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
        stream.hyp = []
        all_streams.append(stream)
        decode_streams.append(stream)

    tail_length = chunk_length + pad_length

    batch_states = stack_states([s.states for s in decode_streams])

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

    results = []
    for idx in range(num_streams):
        stream = all_streams[idx]
        text = token_ids_to_text(stream.hyp, token_table) if stream.hyp else ""
        results.append(
            {
                "filename": args.sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )
    return results


def infer_onnx(args) -> List[dict]:
    """ONNX non-streaming transducer inference."""

    model = OnnxTransducerModel(
        args.encoder,
        args.decoder,
        args.joiner,
    )
    features, feature_lengths = extract_features(args)

    encoder_out, encoder_out_lens = model.run_encoder(features, feature_lengths)

    hyps = greedy_search(model, encoder_out, encoder_out_lens)

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def infer_onnx_ctc(args) -> List[dict]:
    """ONNX non-streaming CTC inference."""
    model = OnnxCtcModel(args.model)

    features, feature_lengths = extract_features(args)
    features = features.cpu()
    feature_lengths = feature_lengths.cpu()

    log_probs, log_probs_len = model(features, feature_lengths)

    hyps = ctc_greedy_search(log_probs, log_probs_len)

    token_table = SymbolTable.from_file(args.tokens)
    durations = get_audio_durations(args.sound_files)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def infer_onnx_streaming(args) -> List[dict]:
    """ONNX streaming transducer inference with DecodeStream.

    Processes each utterance sequentially (ONNX streaming models are batch_size=1).
    Uses DecodeStream to manage features and streaming_greedy_search for decoding.
    Encoder states are stored in stream.states and restored before each chunk.
    """
    model = OnnxStreamingTransducerModel(
        args.encoder,
        args.decoder,
        args.joiner,
    )

    features, feature_lengths = extract_features(args)

    token_table = SymbolTable.from_file(args.tokens)
    context_size = model.context_size
    segment = model.segment
    offset = model.offset
    durations = get_audio_durations(args.sound_files)

    params = AttributeDict(
        {
            "blank_id": getattr(model, "blank_id", 0),
            "context_size": context_size,
            "decoding_method": "greedy_search",
        }
    )

    results = []
    for idx in range(len(args.sound_files)):
        model.reset_states()
        stream = DecodeStream(
            params=params,
            utt_id=args.sound_files[idx],
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
                "filename": args.sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
        )
    return results


def infer_onnx_streaming_ctc(args) -> List[dict]:
    """ONNX streaming CTC inference with DecodeStream.

    Processes each utterance sequentially (ONNX streaming models are batch_size=1).
    Uses DecodeStream to manage features and streaming_ctc_greedy_search for decoding.
    Encoder states are stored in stream.states and restored before each chunk.
    """
    model = OnnxStreamingCtcModel(args.model)

    features, feature_lengths = extract_features(args)

    token_table = SymbolTable.from_file(args.tokens)
    segment = model.segment
    offset = model.offset
    durations = get_audio_durations(args.sound_files)

    params = AttributeDict(
        {
            "blank_id": 0,
            "context_size": 1,
            "decoding_method": "greedy_search",
        }
    )

    results = []
    for idx in range(len(args.sound_files)):
        model.reset_states()

        stream = DecodeStream(
            params=params,
            utt_id=args.sound_files[idx],
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
                "filename": args.sound_files[idx],
                "text": text,
                "duration": durations[idx],
            }
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

def download_hf_model(repo_id: str) -> Path:
    """Download a HuggingFace model repo using huggingface_hub.

    Returns the local cache directory path where the repo is downloaded.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id)
    logging.info(f"Downloaded HuggingFace model '{repo_id}' to {local_dir}")
    return Path(local_dir)


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
            if args.ctc:
                results = infer_jit_streaming_ctc(args)
            else:
                results = infer_jit_streaming(args)
        elif args.ctc:
            results = infer_jit_ctc(args)
        else:
            results = infer_jit(args)
    elif args.model_type == "onnx":
        args.device = torch.device("cpu")  # ONNX models run on CPU
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
