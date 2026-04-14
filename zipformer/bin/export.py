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
Unified export script for Zipformer models.

Supports exporting to PyTorch state_dict, TorchScript, and ONNX formats,
for both streaming and non-streaming models, with transducer or CTC heads.

Usage examples:

(1) Export PyTorch state_dict (non-streaming):

  python export.py \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9

(2) Export TorchScript (non-streaming):

  python export.py \\
    --export-type torch --jit true \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9

(3) Export TorchScript (streaming):

  python export.py \\
    --export-type torch --jit true --causal true \\
    --chunk-size 16 --left-context-frames 128 \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9

(4) Export ONNX non-streaming transducer:

  python export.py \\
    --export-type onnx \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9 --fp16 true

(5) Export ONNX non-streaming CTC:

  python export.py \\
    --export-type onnx --ctc true \\
    --use-transducer 0 --use-ctc 1 \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9

(6) Export ONNX streaming transducer:

  python export.py \\
    --export-type onnx --streaming true \\
    --causal true --chunk-size 16 --left-context-frames 128 \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9 --fp16 true

(7) Export ONNX streaming CTC:

  python export.py \\
    --export-type onnx --streaming true --ctc true \\
    --causal true --chunk-size 16 --left-context-frames 128 \\
    --use-ctc 1 \\
    --exp-dir ./exp --tokens data/lang_bpe_500/tokens.txt \\
    --epoch 30 --avg 9
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import k2
    from scaling_converter import convert_scaled_to_non_scaled
    from train import add_model_arguments, get_model, get_params

    from icefall.checkpoint import (
        average_checkpoints,
        average_checkpoints_with_averaged_model,
        find_checkpoints,
        load_checkpoint,
    )
    from icefall.utils import make_pad_mask, num_tokens, str2bool
except ImportError:
    # Allow --help to work even when dependencies are not installed
    from icefall.utils import str2bool

    def add_model_arguments(parser):
        pass


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--export-type",
        type=str,
        default="torch",
        choices=["torch", "onnx"],
        help="Export format: 'torch' for state_dict/TorchScript, 'onnx' for ONNX.",
    )

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=False,
        help="Whether to export a streaming model (for ONNX export).",
    )

    parser.add_argument(
        "--ctc",
        type=str2bool,
        default=False,
        help="Whether to export the CTC head instead of transducer (for ONNX export).",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script (torch export only).
        It will generate a file named jit_script.pt.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="Whether to also export models in fp16 (ONNX export only).",
    )

    parser.add_argument(
        "--dynamic-batch",
        type=int,
        default=1,
        help="1 to support dynamic batch size. 0 to support only batch size == 1 "
        "(streaming ONNX export only).",
    )

    parser.add_argument(
        "--enable-int8-quantization",
        type=int,
        default=1,
        help="1 to also export int8 ONNX models (streaming ONNX export only).",
    )

    parser.add_argument(
        "--use-whisper-features",
        type=str2bool,
        default=False,
        help="True to use whisper features. Must match the one used in training "
        "(streaming ONNX export only).",
    )

    parser.add_argument(
        "--use-external-data",
        type=str2bool,
        default=False,
        help="Set it to true for model file size > 2GB (streaming ONNX export only).",
    )

    add_model_arguments(parser)

    return parser


# ==============================================================================
# Shared ONNX utilities
# ==============================================================================


def add_meta_data(
    filename: str, meta_data: Dict[str, str], use_external_data: bool = False
):
    """Add meta data to an ONNX model. It is changed in-place."""
    import onnx

    filename = str(filename)
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    if use_external_data:
        external_filename = Path(filename).stem
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    import onnxmltools
    from onnxmltools.utils.float16_converter import convert_float_to_float16

    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def export_onnx_fp16_large_2gb(onnx_fp32_path, onnx_fp16_path):
    import onnxmltools
    from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path

    onnx_fp16_model = convert_float_to_float16_model_path(
        onnx_fp32_path, keep_io_types=True
    )
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def build_streaming_inputs_outputs(
    tensors, i, inputs, outputs, input_names, output_names
):
    """Build dynamic axes for streaming encoder states."""
    assert len(tensors) == 6, len(tensors)

    # (downsample_left, batch_size, key_dim)
    name = f"cached_key_{i}"
    logging.info(f"{name}.shape: {tensors[0].shape}")
    inputs[name] = {1: "N"}
    outputs[f"new_{name}"] = {1: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (1, batch_size, downsample_left, nonlin_attn_head_dim)
    name = f"cached_nonlin_attn_{i}"
    logging.info(f"{name}.shape: {tensors[1].shape}")
    inputs[name] = {1: "N"}
    outputs[f"new_{name}"] = {1: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (downsample_left, batch_size, value_dim)
    name = f"cached_val1_{i}"
    logging.info(f"{name}.shape: {tensors[2].shape}")
    inputs[name] = {1: "N"}
    outputs[f"new_{name}"] = {1: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (downsample_left, batch_size, value_dim)
    name = f"cached_val2_{i}"
    logging.info(f"{name}.shape: {tensors[3].shape}")
    inputs[name] = {1: "N"}
    outputs[f"new_{name}"] = {1: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size, embed_dim, conv_left_pad)
    name = f"cached_conv1_{i}"
    logging.info(f"{name}.shape: {tensors[4].shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size, embed_dim, conv_left_pad)
    name = f"cached_conv2_{i}"
    logging.info(f"{name}.shape: {tensors[5].shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")


def get_streaming_meta_data(encoder_model, comment, use_whisper_features=False):
    """Build metadata dict for streaming ONNX models."""
    ds = encoder_model.encoder.downsampling_factor
    left_context_len = encoder_model.left_context_len
    left_context_len_list = [left_context_len // k for k in ds]

    meta_data = {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": comment,
        "decode_chunk_len": str(encoder_model.chunk_size * 2),
        "T": str(encoder_model.chunk_size * 2 + encoder_model.pad_length),
        "num_encoder_layers": ",".join(
            map(str, encoder_model.encoder.num_encoder_layers)
        ),
        "encoder_dims": ",".join(map(str, encoder_model.encoder.encoder_dim)),
        "cnn_module_kernels": ",".join(
            map(str, encoder_model.encoder.cnn_module_kernel)
        ),
        "left_context_len": ",".join(map(str, left_context_len_list)),
        "query_head_dims": ",".join(map(str, encoder_model.encoder.query_head_dim)),
        "value_head_dims": ",".join(map(str, encoder_model.encoder.value_head_dim)),
        "num_heads": ",".join(map(str, encoder_model.encoder.num_heads)),
    }
    if use_whisper_features:
        meta_data["feature"] = "whisper"
    return meta_data


def export_streaming_encoder_onnx(
    encoder_model,
    encoder_filename,
    opset_version,
    feature_dim,
    dynamic_batch,
    use_external_data,
    output_name,
    meta_data,
):
    """Shared logic for exporting streaming encoder (transducer or CTC) to ONNX."""
    encoder_model.encoder.__class__.forward = (
        encoder_model.encoder.__class__.streaming_forward
    )

    T = encoder_model.chunk_size * 2 + encoder_model.pad_length
    x = torch.rand(1, T, feature_dim, dtype=torch.float32)
    init_state = encoder_model.get_init_states()

    logging.info(f"num_encoders: {len(encoder_model.encoder.encoder_dim)}")
    logging.info(f"len(init_state): {len(init_state)}")

    inputs = {}
    input_names = ["x"]
    outputs = {}
    output_names = [output_name]

    for i in range(len(init_state[:-2]) // 6):
        build_streaming_inputs_outputs(
            init_state[i * 6 : (i + 1) * 6],
            i,
            inputs,
            outputs,
            input_names,
            output_names,
        )

    # (batch_size, channels, left_pad, freq)
    embed_states = init_state[-2]
    name = "embed_states"
    logging.info(f"{name}.shape: {embed_states.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size,)
    processed_lens = init_state[-1]
    name = "processed_lens"
    logging.info(f"{name}.shape: {processed_lens.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    logging.info(f"input_names: {input_names}")
    logging.info(f"output_names: {output_names}")

    torch.onnx.export(
        encoder_model,
        (x, init_state),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x": {0: "N"},
            output_name: {0: "N"},
            **inputs,
            **outputs,
        }
        if dynamic_batch
        else {},
    )

    add_meta_data(
        filename=encoder_filename,
        meta_data=meta_data,
        use_external_data=use_external_data,
    )


# ==============================================================================
# Torch Export (state_dict / TorchScript)
# ==============================================================================


class EncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed (non-streaming JIT export)."""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(features, feature_lengths)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        return encoder_out, encoder_out_lens


class StreamingEncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed (streaming JIT export)."""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        assert len(encoder.chunk_size) == 1, encoder.chunk_size
        assert len(encoder.left_context_frames) == 1, encoder.left_context_frames
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3
        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        chunk_size = self.chunk_size
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=features,
            x_lens=feature_lengths,
            cached_left_pad=cached_embed_left_pad,
        )
        assert x.size(1) == chunk_size, (x.size(1), chunk_size)

        src_key_padding_mask = make_pad_mask(x_lens)

        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        new_processed_lens = processed_lens + x_lens

        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        encoder_states = states[:-2]

        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]
        return encoder_out, encoder_out_lens, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        states = self.encoder.get_init_states(batch_size, device)
        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)
        processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        states.append(processed_lens)
        return states


def export_torch(params, model):
    """Export model as PyTorch state_dict or TorchScript."""
    if params.jit is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)

        if params.causal:
            model.encoder = StreamingEncoderModel(model.encoder, model.encoder_embed)
            chunk_size = model.encoder.chunk_size
            left_context_len = model.encoder.left_context_len
            filename = f"jit_script_chunk_{chunk_size}_left_{left_context_len}.pt"
        else:
            model.encoder = EncoderModel(model.encoder, model.encoder_embed)
            filename = "jit_script.pt"

        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        model.save(str(params.exp_dir / filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torchscript. Export model.state_dict()")
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


# ==============================================================================
# ONNX Non-streaming Transducer
# ==============================================================================


class OnnxEncoder(nn.Module):
    """A wrapper for Zipformer and the encoder_proj from the joiner (non-streaming)."""

    def __init__(self, encoder, encoder_embed, encoder_proj):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens, x.shape[1])
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)
        return encoder_out, encoder_out_lens


class OnnxDecoder(nn.Module):
    """A wrapper for Decoder and the decoder_proj from the joiner."""

    def __init__(self, decoder, decoder_proj):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        need_pad = False
        decoder_output = self.decoder(y, need_pad=need_pad)
        decoder_output = decoder_output.squeeze(1)
        output = self.decoder_proj(decoder_output)
        return output


class OnnxJoiner(nn.Module):
    """A wrapper for the joiner."""

    def __init__(self, output_linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        logit = encoder_out + decoder_out
        logit = self.output_linear(torch.tanh(logit))
        return logit


def _export_encoder_model_onnx(encoder_model, encoder_filename, opset_version=11):
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    encoder_model = torch.jit.trace(encoder_model, (x, x_lens))

    torch.onnx.export(
        encoder_model,
        (x, x_lens),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "encoder_out": {0: "N", 1: "T"},
            "encoder_out_lens": {0: "N"},
        },
    )

    meta_data = {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "non-streaming zipformer2",
    }
    logging.info(f"meta_data: {meta_data}")
    add_meta_data(filename=encoder_filename, meta_data=meta_data)


def _export_decoder_model_onnx(
    decoder_model, decoder_filename, opset_version=11, dynamic_batch=True
):
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size

    y = torch.zeros(10, context_size, dtype=torch.int64)
    decoder_model = torch.jit.script(decoder_model)
    torch.onnx.export(
        decoder_model,
        y,
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        }
        if dynamic_batch
        else {},
    )

    meta_data = {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def _export_joiner_model_onnx(
    joiner_model, joiner_filename, opset_version=11, dynamic_batch=True
):
    joiner_dim = joiner_model.output_linear.weight.shape[1]
    logging.info(f"joiner dim: {joiner_dim}")

    projected_encoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        }
        if dynamic_batch
        else {},
    )
    meta_data = {"joiner_dim": str(joiner_dim)}
    add_meta_data(filename=joiner_filename, meta_data=meta_data)


def export_onnx_transducer(params, model):
    """Export non-streaming transducer model to ONNX."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    encoder = OnnxEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )
    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )
    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"
    suffix += f"-avg-{params.avg}"

    opset_version = 13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / f"encoder-{suffix}.onnx"
    _export_encoder_model_onnx(encoder, encoder_filename, opset_version=opset_version)
    logging.info(f"Exported encoder to {encoder_filename}")

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / f"decoder-{suffix}.onnx"
    _export_decoder_model_onnx(decoder, decoder_filename, opset_version=opset_version)
    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / f"joiner-{suffix}.onnx"
    _export_joiner_model_onnx(joiner, joiner_filename, opset_version=opset_version)
    logging.info(f"Exported joiner to {joiner_filename}")

    if params.fp16:
        logging.info("Generate fp16 models")
        encoder_filename_fp16 = params.exp_dir / f"encoder-{suffix}.fp16.onnx"
        export_onnx_fp16(encoder_filename, encoder_filename_fp16)
        decoder_filename_fp16 = params.exp_dir / f"decoder-{suffix}.fp16.onnx"
        export_onnx_fp16(decoder_filename, decoder_filename_fp16)
        joiner_filename_fp16 = params.exp_dir / f"joiner-{suffix}.fp16.onnx"
        export_onnx_fp16(joiner_filename, joiner_filename_fp16)

    logging.info("Generate int8 quantization models")

    encoder_filename_int8 = params.exp_dir / f"encoder-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=encoder_filename,
        model_output=encoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    decoder_filename_int8 = params.exp_dir / f"decoder-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=decoder_filename,
        model_output=decoder_filename_int8,
        op_types_to_quantize=["MatMul", "Gather"],
        weight_type=QuantType.QInt8,
    )

    joiner_filename_int8 = params.exp_dir / f"joiner-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=joiner_filename,
        model_output=joiner_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


# ==============================================================================
# ONNX Non-streaming CTC
# ==============================================================================


class OnnxCtcModel(nn.Module):
    """A wrapper for encoder_embed, Zipformer, and ctc_output layer (non-streaming)."""

    def __init__(self, encoder, encoder_embed, ctc_output):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.ctc_output = ctc_output

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)
        encoder_out, log_probs_len = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        log_probs = self.ctc_output(encoder_out)
        return log_probs, log_probs_len


def _export_ctc_model_onnx(model, filename, opset_version=11):
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    model = torch.jit.trace(model, (x, x_lens))

    torch.onnx.export(
        model,
        (x, x_lens),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["log_probs", "log_probs_len"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "log_probs": {0: "N", 1: "T"},
            "log_probs_len": {0: "N"},
        },
    )

    meta_data = {
        "model_type": "zipformer2_ctc",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "non-streaming zipformer2 CTC",
    }
    logging.info(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)


def export_onnx_ctc(params, model):
    """Export non-streaming CTC model to ONNX."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ctc_model = OnnxCtcModel(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        ctc_output=model.ctc_output,
    )

    num_param = sum([p.numel() for p in ctc_model.parameters()])
    logging.info(f"num parameters: {num_param}")

    opset_version = 13

    logging.info("Exporting ctc model")
    filename = params.exp_dir / "model.onnx"
    _export_ctc_model_onnx(ctc_model, filename, opset_version=opset_version)
    logging.info(f"Exported to {filename}")

    logging.info("Generate int8 quantization models")
    filename_int8 = params.exp_dir / "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    if params.fp16:
        filename_fp16 = params.exp_dir / "model.fp16.onnx"
        export_onnx_fp16(filename, filename_fp16)


# ==============================================================================
# ONNX Streaming Transducer
# ==============================================================================


class OnnxStreamingEncoder(nn.Module):
    """A wrapper for Zipformer and the encoder_proj from the joiner (streaming)."""

    def __init__(self, encoder, encoder_embed, encoder_proj):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3

    def forward(
        self, x: torch.Tensor, states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        N = x.size(0)
        T = self.chunk_size * 2 + self.pad_length
        x_lens = torch.tensor([T] * N, device=x.device)
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=cached_embed_left_pad,
        )
        assert x.size(1) == self.chunk_size, (x.size(1), self.chunk_size)

        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool)

        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        new_processed_lens = processed_lens + x_lens
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        encoder_states = states[:-2]
        logging.info(f"len_encoder_states={len(encoder_states)}")
        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]
        return encoder_out, new_states

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        states = self.encoder.get_init_states(batch_size, device)
        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)
        processed_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        states.append(processed_lens)
        return states


def export_onnx_streaming_transducer(params, model):
    """Export streaming transducer model to ONNX."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    encoder = OnnxStreamingEncoder(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )
    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )
    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"
    suffix += f"-avg-{params.avg}"
    suffix += f"-chunk-{params.chunk_size}"
    suffix += f"-left-{params.left_context_frames}"

    opset_version = 13
    dynamic_batch = params.dynamic_batch == 1

    meta_data = get_streaming_meta_data(
        encoder,
        "streaming zipformer2",
        use_whisper_features=params.use_whisper_features,
    )
    logging.info(f"meta_data: {meta_data}")

    logging.info("Exporting encoder")
    if params.use_external_data:
        encoder_filename = f"encoder-{suffix}.onnx"
    else:
        encoder_filename = params.exp_dir / f"encoder-{suffix}.onnx"
    export_streaming_encoder_onnx(
        encoder,
        str(encoder_filename),
        opset_version=opset_version,
        feature_dim=params.feature_dim,
        dynamic_batch=dynamic_batch,
        use_external_data=params.use_external_data,
        output_name="encoder_out",
        meta_data=meta_data,
    )
    logging.info(f"Exported encoder to {encoder_filename}")

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / f"decoder-{suffix}.onnx"
    _export_decoder_model_onnx(
        decoder,
        decoder_filename,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
    )
    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / f"joiner-{suffix}.onnx"
    _export_joiner_model_onnx(
        joiner,
        joiner_filename,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
    )
    logging.info(f"Exported joiner to {joiner_filename}")

    if params.fp16:
        logging.info("Generate fp16 models")
        if params.use_external_data:
            encoder_filename_fp16 = f"encoder-{suffix}.fp16.onnx"
            export_onnx_fp16_large_2gb(encoder_filename, encoder_filename_fp16)
        else:
            encoder_filename_fp16 = params.exp_dir / f"encoder-{suffix}.fp16.onnx"
            export_onnx_fp16(encoder_filename, encoder_filename_fp16)
        decoder_filename_fp16 = params.exp_dir / f"decoder-{suffix}.fp16.onnx"
        export_onnx_fp16(decoder_filename, decoder_filename_fp16)
        joiner_filename_fp16 = params.exp_dir / f"joiner-{suffix}.fp16.onnx"
        export_onnx_fp16(joiner_filename, joiner_filename_fp16)

    if params.enable_int8_quantization:
        logging.info("Generate int8 quantization models")

        if params.use_external_data:
            encoder_filename_int8 = f"encoder-{suffix}.int8.onnx"
        else:
            encoder_filename_int8 = params.exp_dir / f"encoder-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=encoder_filename,
            model_output=encoder_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

        decoder_filename_int8 = params.exp_dir / f"decoder-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=decoder_filename,
            model_output=decoder_filename_int8,
            op_types_to_quantize=["MatMul", "Gather"],
            weight_type=QuantType.QInt8,
        )

        joiner_filename_int8 = params.exp_dir / f"joiner-{suffix}.int8.onnx"
        quantize_dynamic(
            model_input=joiner_filename,
            model_output=joiner_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )


# ==============================================================================
# ONNX Streaming CTC
# ==============================================================================


class OnnxStreamingCtcModel(nn.Module):
    """A wrapper for Zipformer and the ctc_head (streaming)."""

    def __init__(self, encoder, encoder_embed, ctc_output):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.ctc_output = ctc_output
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3

    def forward(
        self, x: torch.Tensor, states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        N = x.size(0)
        T = self.chunk_size * 2 + self.pad_length
        x_lens = torch.tensor([T] * N, device=x.device)
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=cached_embed_left_pad,
        )
        assert x.size(1) == self.chunk_size, (x.size(1), self.chunk_size)

        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool)

        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        new_processed_lens = processed_lens + x_lens
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        encoder_states = states[:-2]
        logging.info(f"len_encoder_states={len(encoder_states)}")
        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.ctc_output(encoder_out)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]
        return encoder_out, new_states

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        states = self.encoder.get_init_states(batch_size, device)
        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)
        processed_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        states.append(processed_lens)
        return states


def export_onnx_streaming_ctc(params, model):
    """Export streaming CTC model to ONNX."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ctc_model = OnnxStreamingCtcModel(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        ctc_output=model.ctc_output,
    )

    total_num_param = sum([p.numel() for p in ctc_model.parameters()])
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"
    suffix += f"-avg-{params.avg}"
    suffix += f"-chunk-{params.chunk_size}"
    suffix += f"-left-{params.left_context_frames}"

    opset_version = 13
    dynamic_batch = params.dynamic_batch == 1

    meta_data = get_streaming_meta_data(
        ctc_model,
        "streaming ctc zipformer2",
        use_whisper_features=params.use_whisper_features,
    )
    logging.info(f"meta_data: {meta_data}")

    logging.info("Exporting model")
    if params.use_external_data:
        model_filename = f"ctc-{suffix}.onnx"
    else:
        model_filename = params.exp_dir / f"ctc-{suffix}.onnx"

    export_streaming_encoder_onnx(
        ctc_model,
        str(model_filename),
        opset_version=opset_version,
        feature_dim=params.feature_dim,
        dynamic_batch=dynamic_batch,
        use_external_data=params.use_external_data,
        output_name="log_probs",
        meta_data=meta_data,
    )
    logging.info(f"Exported model to {model_filename}")

    if params.enable_int8_quantization:
        logging.info("Generate int8 quantization models")

        if params.use_external_data:
            model_filename_int8 = f"ctc-{suffix}.int8.onnx"
        else:
            model_filename_int8 = params.exp_dir / f"ctc-{suffix}.int8.onnx"

        quantize_dynamic(
            model_input=model_filename,
            model_output=model_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

    if params.fp16:
        if params.use_external_data:
            model_filename_fp16 = f"ctc-{suffix}.fp16.onnx"
            export_onnx_fp16_large_2gb(model_filename, model_filename_fp16)
        else:
            model_filename_fp16 = params.exp_dir / f"ctc-{suffix}.fp16.onnx"
            export_onnx_fp16(model_filename, model_filename_fp16)


# ==============================================================================
# Shared checkpoint loading
# ==============================================================================


def load_model_checkpoint(params, model, device, strict=True):
    """Load and average checkpoints into model."""
    model.to(device)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=strict
            )
        elif params.avg == 1:
            load_checkpoint(
                f"{params.exp_dir}/epoch-{params.epoch}.pt", model, strict=strict
            )
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=strict
            )
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=strict,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=strict,
            )


# ==============================================================================
# Main
# ==============================================================================


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # Determine device
    if params.export_type == "torch":
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    # Load token table
    token_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = token_table["<blk>"]
    params.vocab_size = num_tokens(token_table) + 1

    if params.export_type == "torch":
        params.sos_id = params.eos_id = token_table["<sos/eos>"]

    logging.info(params)

    # Create model
    logging.info("About to create model")
    model = get_model(params)

    # Load checkpoint
    # Use strict=False for CTC-only export since model may have transducer components
    strict = not params.ctc
    load_model_checkpoint(params, model, device, strict=strict)

    # Move to CPU and eval
    model.to("cpu")
    model.eval()

    # Route to the appropriate export function
    if params.export_type == "torch":
        export_torch(params, model)
    elif params.export_type == "onnx":
        is_onnx = not params.streaming
        convert_scaled_to_non_scaled(model, inplace=True, is_onnx=is_onnx)

        if params.streaming:
            if params.ctc:
                export_onnx_streaming_ctc(params, model)
            else:
                export_onnx_streaming_transducer(params, model)
        else:
            if params.ctc:
                export_onnx_ctc(params, model)
            else:
                export_onnx_transducer(params, model)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
