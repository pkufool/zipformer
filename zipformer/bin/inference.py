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
"""

import argparse
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from icefall.utils import str2bool


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
        choices=["jit", "onnx"],
        help="Model format: 'jit' for TorchScript, 'onnx' for ONNX.",
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
        "--nn-model-filename",
        type=str,
        default="",
        help="Path to the TorchScript model (for --model-type jit).",
    )

    # ONNX transducer model paths
    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        default="",
        help="Path to the encoder ONNX model (for --model-type onnx, transducer).",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        default="",
        help="Path to the decoder ONNX model (for --model-type onnx, transducer).",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        default="",
        help="Path to the joiner ONNX model (for --model-type onnx, transducer).",
    )

    # ONNX CTC model path
    parser.add_argument(
        "--nn-model",
        type=str,
        default="",
        help="Path to the single ONNX model (for --model-type onnx, CTC).",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt.",
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
        nargs="+",
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
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        )
        ans.append(wave[0].contiguous())
    return ans


def get_audio_durations(filenames: List[str], sample_rate: int) -> List[float]:
    """Get duration in seconds for each audio file."""
    durations = []
    for f in filenames:
        info = torchaudio.info(f)
        durations.append(info.num_frames / info.sample_rate)
    return durations


def create_fbank(sample_rate: int = 16000, device: str = "cpu"):
    """Create a non-streaming Fbank feature extractor."""
    import kaldifeat

    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return kaldifeat.Fbank(opts)


def create_streaming_fbank(sample_rate: int = 16000):
    """Create a CPU streaming OnlineFbank feature extractor."""
    from kaldifeat import FbankOptions, OnlineFbank

    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return OnlineFbank(opts)


# ==============================================================================
# Token decoding
# ==============================================================================


def load_token_table(tokens_path: str):
    """Load token table from tokens.txt using k2.SymbolTable."""
    import k2

    return k2.SymbolTable.from_file(tokens_path)


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
# ONNX Model Wrappers
# ==============================================================================


class OnnxTransducerModel:
    """Non-streaming ONNX transducer model (encoder + decoder + joiner)."""

    def __init__(self, encoder_filename, decoder_filename, joiner_filename):
        import onnxruntime as ort

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.encoder = ort.InferenceSession(
            encoder_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            decoder_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.joiner = ort.InferenceSession(
            joiner_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

    def run_encoder(self, x, x_lens):
        out = self.encoder.run(
            [self.encoder.get_outputs()[0].name, self.encoder.get_outputs()[1].name],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])

    def run_decoder(self, decoder_input):
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]
        return torch.from_numpy(out)

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]
        return torch.from_numpy(out)


class OnnxCtcModel:
    """Non-streaming ONNX CTC model."""

    def __init__(self, nn_model):
        import onnxruntime as ort

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.model = ort.InferenceSession(
            nn_model,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, x, x_lens):
        out = self.model.run(
            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


class OnnxStreamingTransducerModel:
    """Streaming ONNX transducer model with state management."""

    def __init__(self, encoder_filename, decoder_filename, joiner_filename):
        import onnxruntime as ort

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            decoder_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.joiner = ort.InferenceSession(
            joiner_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

        self._init_encoder_states()

    def _init_encoder_states(self, batch_size=1):
        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.segment = int(meta["T"])
        self.offset = int(meta["decode_chunk_len"])

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(meta["num_encoder_layers"])
        encoder_dims = to_int_list(meta["encoder_dims"])
        cnn_module_kernels = to_int_list(meta["cnn_module_kernels"])
        left_context_len = to_int_list(meta["left_context_len"])
        query_head_dims = to_int_list(meta["query_head_dims"])
        value_head_dims = to_int_list(meta["value_head_dims"])
        num_heads = to_int_list(meta["num_heads"])

        self.states = []
        for i in range(len(num_encoder_layers)):
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2

            for _ in range(num_encoder_layers[i]):
                self.states += [
                    np.zeros(
                        (left_context_len[i], batch_size, key_dim), dtype=np.float32
                    ),
                    np.zeros(
                        (1, batch_size, left_context_len[i], nonlin_attn_head_dim),
                        dtype=np.float32,
                    ),
                    np.zeros(
                        (left_context_len[i], batch_size, value_dim), dtype=np.float32
                    ),
                    np.zeros(
                        (left_context_len[i], batch_size, value_dim), dtype=np.float32
                    ),
                    np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32),
                    np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32),
                ]
        self.states.append(np.zeros((batch_size, 128, 3, 19), dtype=np.float32))
        self.states.append(np.zeros(batch_size, dtype=np.int64))

    def reset_states(self):
        self._init_encoder_states()

    def _build_encoder_io(self, x):
        encoder_input = {"x": x.numpy()}
        encoder_output = ["encoder_out"]

        for i in range(len(self.states[:-2]) // 6):
            tensors = self.states[i * 6 : (i + 1) * 6]
            for j, prefix in enumerate(
                [
                    "cached_key",
                    "cached_nonlin_attn",
                    "cached_val1",
                    "cached_val2",
                    "cached_conv1",
                    "cached_conv2",
                ]
            ):
                name = f"{prefix}_{i}"
                encoder_input[name] = tensors[j]
                encoder_output.append(f"new_{name}")

        encoder_input["embed_states"] = self.states[-2]
        encoder_output.append("new_embed_states")
        encoder_input["processed_lens"] = self.states[-1]
        encoder_output.append("new_processed_lens")

        return encoder_input, encoder_output

    def run_encoder(self, x):
        encoder_input, encoder_output_names = self._build_encoder_io(x)
        out = self.encoder.run(encoder_output_names, encoder_input)
        self.states = out[1:]
        return torch.from_numpy(out[0])

    def run_decoder(self, decoder_input):
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]
        return torch.from_numpy(out)

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]
        return torch.from_numpy(out)


class OnnxStreamingCtcModel:
    """Streaming ONNX CTC model with state management."""

    def __init__(self, model_filename):
        import onnxruntime as ort

        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.model = ort.InferenceSession(
            model_filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self._init_states()

    def _init_states(self, batch_size=1):
        meta = self.model.get_modelmeta().custom_metadata_map
        self.segment = int(meta["T"])
        self.offset = int(meta["decode_chunk_len"])

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(meta["num_encoder_layers"])
        encoder_dims = to_int_list(meta["encoder_dims"])
        cnn_module_kernels = to_int_list(meta["cnn_module_kernels"])
        left_context_len = to_int_list(meta["left_context_len"])
        query_head_dims = to_int_list(meta["query_head_dims"])
        value_head_dims = to_int_list(meta["value_head_dims"])
        num_heads = to_int_list(meta["num_heads"])

        self.states = []
        for i in range(len(num_encoder_layers)):
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2

            for _ in range(num_encoder_layers[i]):
                self.states += [
                    np.zeros(
                        (left_context_len[i], batch_size, key_dim), dtype=np.float32
                    ),
                    np.zeros(
                        (1, batch_size, left_context_len[i], nonlin_attn_head_dim),
                        dtype=np.float32,
                    ),
                    np.zeros(
                        (left_context_len[i], batch_size, value_dim), dtype=np.float32
                    ),
                    np.zeros(
                        (left_context_len[i], batch_size, value_dim), dtype=np.float32
                    ),
                    np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32),
                    np.zeros((batch_size, embed_dim, conv_left_pad), dtype=np.float32),
                ]
        self.states.append(np.zeros((batch_size, 128, 3, 19), dtype=np.float32))
        self.states.append(np.zeros(batch_size, dtype=np.int64))

    def reset_states(self):
        self._init_states()

    def _build_model_io(self, x):
        model_input = {"x": x.numpy()}
        model_output = ["log_probs"]

        for i in range(len(self.states[:-2]) // 6):
            tensors = self.states[i * 6 : (i + 1) * 6]
            for j, prefix in enumerate(
                [
                    "cached_key",
                    "cached_nonlin_attn",
                    "cached_val1",
                    "cached_val2",
                    "cached_conv1",
                    "cached_conv2",
                ]
            ):
                name = f"{prefix}_{i}"
                model_input[name] = tensors[j]
                model_output.append(f"new_{name}")

        model_input["embed_states"] = self.states[-2]
        model_output.append("new_embed_states")
        model_input["processed_lens"] = self.states[-1]
        model_output.append("new_processed_lens")

        return model_input, model_output

    def __call__(self, x):
        model_input, model_output_names = self._build_model_io(x)
        out = self.model.run(model_output_names, model_input)
        self.states = out[1:]
        return torch.from_numpy(out[0])


# ==============================================================================
# Greedy search implementations
# ==============================================================================


def greedy_search_transducer_batch(model, encoder_out, encoder_out_lens):
    """Batch greedy search for non-streaming transducer (ONNX)."""
    assert encoder_out.ndim == 3

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = 0
    N = encoder_out.size(0)
    context_size = model.context_size
    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(hyps, dtype=torch.int64)
    decoder_out = model.run_decoder(decoder_input)

    offset = 0
    for batch_size in packed.batch_sizes.tolist():
        start = offset
        end = offset + batch_size
        current_encoder_out = packed.data[start:end]
        offset = end

        decoder_out = decoder_out[:batch_size]
        logits = model.run_joiner(current_encoder_out, decoder_out)

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(decoder_input, dtype=torch.int64)
            decoder_out = model.run_decoder(decoder_input)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans


def greedy_search_transducer_batch_jit(model, encoder_out, encoder_out_lens):
    """Batch greedy search for non-streaming transducer (JIT)."""
    assert encoder_out.ndim == 3

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = encoder_out.device
    blank_id = model.decoder.blank_id
    N = encoder_out.size(0)
    context_size = model.decoder.context_size

    hyps = [[blank_id] * context_size for _ in range(N)]
    decoder_input = torch.tensor(hyps, device=device, dtype=torch.int64)
    decoder_out = model.decoder(decoder_input, need_pad=torch.tensor([False])).squeeze(
        1
    )

    offset = 0
    for batch_size in packed.batch_sizes.tolist():
        start = offset
        end = offset + batch_size
        current_encoder_out = packed.data[start:end]
        offset = end

        decoder_out = decoder_out[:batch_size]
        logits = model.joiner(current_encoder_out, decoder_out)

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input, device=device, dtype=torch.int64
            )
            decoder_out = model.decoder(
                decoder_input, need_pad=torch.tensor([False])
            ).squeeze(1)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans


def greedy_search_transducer_streaming_onnx(
    model,
    encoder_out,
    context_size,
    decoder_out=None,
    hyp=None,
):
    """Streaming greedy search for ONNX transducer. Processes one chunk."""
    blank_id = 0

    if decoder_out is None:
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)
        decoder_out = model.run_decoder(decoder_input)

    encoder_out = encoder_out.squeeze(0)
    T = encoder_out.size(0)
    for t in range(T):
        cur_encoder_out = encoder_out[t : t + 1]
        joiner_out = model.run_joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = torch.tensor([hyp[-context_size:]], dtype=torch.int64)
            decoder_out = model.run_decoder(decoder_input)

    return hyp, decoder_out


def greedy_search_transducer_streaming_jit(
    decoder,
    joiner,
    encoder_out,
    decoder_out=None,
    hyp=None,
    device=torch.device("cpu"),
):
    """Streaming greedy search for JIT transducer. Processes one chunk."""
    assert encoder_out.ndim == 2
    context_size = decoder.context_size
    blank_id = decoder.blank_id

    if decoder_out is None:
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor(hyp, dtype=torch.int32, device=device).unsqueeze(0)
        decoder_out = decoder(decoder_input, torch.tensor([False])).squeeze(1)

    T = encoder_out.size(0)
    for i in range(T):
        cur_encoder_out = encoder_out[i : i + 1]
        joiner_out = joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()

        if y != blank_id:
            hyp.append(y)
            decoder_input = torch.tensor(
                hyp[-context_size:],
                dtype=torch.int32,
                device=device,
            ).unsqueeze(0)
            decoder_out = decoder(decoder_input, torch.tensor([False])).squeeze(1)

    return hyp, decoder_out


def greedy_search_ctc(log_probs):
    """CTC greedy search: argmax + unique_consecutive + remove blanks."""
    assert log_probs.ndim == 3 and log_probs.shape[0] == 1
    max_indexes = log_probs[0].argmax(dim=1)
    unique_indexes = torch.unique_consecutive(max_indexes)
    blank_id = 0
    unique_indexes = unique_indexes[unique_indexes != blank_id]
    return unique_indexes.tolist()


# ==============================================================================
# Inference functions
# ==============================================================================


def infer_jit(args) -> List[dict]:
    """JIT non-streaming transducer inference."""
    from torch.nn.utils.rnn import pad_sequence

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model = torch.jit.load(args.nn_model_filename)
    model.eval()
    model.to(device)

    fbank = create_fbank(args.sample_rate, device=str(device))
    waves = read_sound_files(args.sound_files, args.sample_rate)
    waves = [w.to(device) for w in waves]

    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, device=device)

    encoder_out, encoder_out_lens = model.encoder(
        features=features,
        feature_lengths=feature_lengths,
    )

    hyps = greedy_search_transducer_batch_jit(model, encoder_out, encoder_out_lens)

    token_table = load_token_table(args.tokens)
    durations = get_audio_durations(args.sound_files, args.sample_rate)

    results = []
    for filename, hyp, dur in zip(args.sound_files, hyps, durations):
        text = token_ids_to_text(hyp, token_table)
        results.append({"filename": filename, "text": text, "duration": dur})
    return results


def infer_jit_streaming(args) -> List[dict]:
    """JIT streaming transducer inference."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model = torch.jit.load(args.nn_model_filename)
    model.eval()
    model.to(device)

    encoder = model.encoder
    decoder = model.decoder
    joiner = model.joiner

    token_table = load_token_table(args.tokens)
    context_size = decoder.context_size

    chunk_length = encoder.chunk_size * 2
    T = chunk_length + encoder.pad_length

    durations = get_audio_durations(args.sound_files, args.sample_rate)
    results = []

    for idx, sound_file in enumerate(args.sound_files):
        online_fbank = create_streaming_fbank(args.sample_rate)
        wave_samples = read_sound_files([sound_file], args.sample_rate)[0]

        states = encoder.get_init_states(device=device)
        tail_padding = torch.zeros(int(0.3 * args.sample_rate), dtype=torch.float32)
        wave_samples = torch.cat([wave_samples, tail_padding])

        chunk = int(0.25 * args.sample_rate)
        num_processed_frames = 0
        hyp = None
        decoder_out = None

        start = 0
        while start < wave_samples.numel():
            end = min(start + chunk, wave_samples.numel())
            samples = wave_samples[start:end]
            start += chunk

            online_fbank.accept_waveform(
                sampling_rate=args.sample_rate, waveform=samples
            )

            while online_fbank.num_frames_ready - num_processed_frames >= T:
                frames = []
                for i in range(T):
                    frames.append(online_fbank.get_frame(num_processed_frames + i))
                frames = torch.cat(frames, dim=0).to(device).unsqueeze(0)
                x_lens = torch.tensor([T], dtype=torch.int32, device=device)
                encoder_out, out_lens, states = encoder(
                    features=frames,
                    feature_lengths=x_lens,
                    states=states,
                )
                num_processed_frames += chunk_length

                hyp, decoder_out = greedy_search_transducer_streaming_jit(
                    decoder,
                    joiner,
                    encoder_out.squeeze(0),
                    decoder_out,
                    hyp,
                    device=device,
                )

        if hyp is not None:
            text = token_ids_to_text(hyp[context_size:], token_table)
        else:
            text = ""
        results.append(
            {"filename": sound_file, "text": text, "duration": durations[idx]}
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

    token_table = load_token_table(args.tokens)
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

    token_table = load_token_table(args.tokens)
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


@torch.no_grad()
def main():
    args = get_parser().parse_args()
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
