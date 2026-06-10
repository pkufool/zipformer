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

(1) Export PyTorch state_dict:

  python export.py \
    --export-type torch \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9

(2) Export TorchScript (non-streaming):

  python export.py \
    --export-type torch \
    --jit true \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9

(3) Export TorchScript (streaming):

  python export.py \
    --export-type torch \
    --jit true \
    --causal true \
    --chunk-size 16 \
    --left-context-frames 128 \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9

(4) Export ONNX non-streaming transducer:

  python export.py \
    --export-type onnx \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9 \
    --fp16 true

(5) Export ONNX non-streaming CTC:

  python export.py \
    --export-type onnx \
    --ctc true \
    --use-transducer 0 \
    --use-ctc 1 \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9

(6) Export ONNX streaming transducer:

  python export.py \
    --export-type onnx \
    --streaming true \
    --causal true \
    --chunk-size 16 \
    --left-context-frames 128 \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9 \
    --fp16 true

(7) Export ONNX streaming CTC:

  python export.py \
    --export-type onnx \
    --streaming true \
    --ctc true \
    --causal true \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 1 \
    --exp-dir ./exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch 30 \
    --avg 9
"""

import argparse
import logging

from pathlib import Path
from typing import Dict, List, Tuple


import torch

import onnx

from onnxconverter_common import float16

from ssentencepiece import Ssentencepiece
from onnxruntime.quantization import QuantType, quantize_dynamic

from zipformer.bin.train import add_model_arguments, get_model, get_params

from zipformer.utils import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
    str2bool, SymbolTable, num_tokens
)

from zipformer.modules.model import (
    EncoderWrapper,
    StreamingEncoderWrapper,
    OnnxEncoderWrapper,
    OnnxDecoderWrapper,
    OnnxJoinerWrapper,
    OnnxCtcWrapper,
    OnnxStreamingEncoderWrapper,
    OnnxStreamingCtcWrapper
)


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
        default="exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens file",
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
        "--enable-int8-quantization",
        type=str2bool,
        default=False,
        help="Whether to also export int8 quantization models (ONNX export only).",
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
        "--use-external-data",
        type=str2bool,
        default=False,
        help="Set it to true for model file size > 2GB (streaming ONNX export only).",
    )

    add_model_arguments(parser)

    return parser


def export_torch(params, model):
    """Export model as PyTorch state_dict or TorchScript."""
    if params.jit is True:
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)

        if params.causal:
            model.encoder = StreamingEncoderWrapper(model.encoder, model.encoder_embed)
            chunk_size = model.encoder.chunk_size
            left_context_len = model.encoder.left_context_len
            filename = f"jit_model_chunk_{chunk_size}_left_{left_context_len}.pt"
        else:
            model.encoder = EncoderWrapper(model.encoder, model.encoder_embed)
            filename = "jit_model.pt"

        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        model.save(str(params.exp_dir / filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torchscript. Export model.state_dict()")
        filename = params.exp_dir / "model.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


# ==============================================================================
# Shared ONNX utilities
# ==============================================================================

def add_meta_data(
    filename: str, meta_data: Dict[str, str], use_external_data: bool = False
):
    """Add meta data to an ONNX model. It is changed in-place."""
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
    onnx_fp32_model = onnx.load_model(onnx_fp32_path)
    onnx_fp16_model = float16.convert_float_to_float16(
        onnx_fp32_model, keep_io_types=True
    )
    onnx.save_model(onnx_fp16_model, onnx_fp16_path)


def export_onnx_fp16_large_2gb(onnx_fp32_path, onnx_fp16_path):
    onnx_fp16_model = float16.convert_float_to_float16_model_path(
        onnx_fp32_path, keep_io_types=True
    )
    onnx.save_model(onnx_fp16_model, onnx_fp16_path)


def _export_encoder_model_onnx(encoder_model, encoder_filename, opset_version=13):
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    # encoder_model = torch.jit.trace(encoder_model, (x, x_lens))
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
    decoder_model, decoder_filename, opset_version=13, dynamic_batch=True
):
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size
    blank_id = decoder_model.decoder.blank_id
    unk_id = getattr(decoder_model, "unk_id", blank_id)

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
        "blank_id": str(blank_id),
        "unk_id": str(unk_id)
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def _export_joiner_model_onnx(
    joiner_model, joiner_filename, opset_version=13, dynamic_batch=True
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

    encoder = OnnxEncoderWrapper(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )
    decoder = OnnxDecoderWrapper(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )
    joiner = OnnxJoinerWrapper(output_linear=model.joiner.output_linear)

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

    # We don't quantize the decoder since it may cause large accuracy drop.
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

def _export_ctc_model_onnx(model, filename, opset_version=11):
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    # model = torch.jit.trace(model, (x, x_lens))

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
        "model_type": "zipformer2_ctc",  # for compatibility with sherpa-onnx
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "non-streaming zipformer2 CTC",
    }
    logging.info(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)


def export_onnx_ctc(params, model):
    """Export non-streaming CTC model to ONNX."""

    ctc_model = OnnxCtcWrapper(
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


def get_streaming_meta_data(encoder_model, comment):
    """Build metadata dict for streaming ONNX models."""
    ds = encoder_model.encoder.downsampling_factor
    left_context_len = encoder_model.left_context_len
    left_context_len_list = [left_context_len // k for k in ds]

    meta_data = {
        "model_type": "zipformer2",  # for compatibility with sherpa-onnx
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
    return meta_data


def export_streaming_encoder_onnx(
    encoder_model: torch.nn.Module,
    encoder_filename: str,
    feature_dim: int,
    dynamic_batch: bool,
    use_external_data: bool,
    output_name: str,
    meta_data: dict,
    opset_version: int = 11,
):
    """Shared logic for exporting streaming encoder (transducer or CTC) to ONNX."""
    encoder_model.encoder.__class__.forward = encoder_model.encoder.__class__.streaming_forward

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


def export_onnx_streaming_transducer(params, model):
    """Export streaming transducer model to ONNX."""

    encoder = OnnxStreamingEncoderWrapper(
        encoder=model.encoder,
        encoder_embed=model.encoder_embed,
        encoder_proj=model.joiner.encoder_proj,
    )
    decoder = OnnxDecoderWrapper(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )
    joiner = OnnxJoinerWrapper(output_linear=model.joiner.output_linear)

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

        # We don't quantize the decoder since it may cause large accuracy drop.

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
def export_onnx_streaming_ctc(params, model):
    """Export streaming CTC model to ONNX."""
    
    ctc_model = OnnxStreamingCtcWrapper(
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
def load_model_checkpoint(params, model, strict=True, device=torch.device("cpu")):
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

    if params.bpe_model is not None and not Path(params.bpe_model).is_file():
        sp = Ssentencepiece(params.bpe_model)
        params.blank_id = sp.piece_to_id("<blk>")
        if sp.piece_to_id("<sos/eos>") != sp.piece_to_id("<unk>"):
            params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
        else:
            params.sos_id = sp.piece_to_id("<sos>")
            params.eos_id = sp.piece_to_id("<eos>")
        params.vocab_size = sp.get_piece_size()
    elif params.tokens is not None and Path(params.tokens).is_file():
        token_table = SymbolTable.from_file(params.tokens)
        params.blank_id = token_table["<blk>"]
        if "<sos/eos>" in token_table:
            params.sos_id = params.eos_id = token_table["<sos/eos>"]
        else:
            params.sos_id = token_table["<sos>"]
            params.eos_id = token_table["<eos>"]
        params.vocab_size = num_tokens(token_table)
    else:
        raise ValueError(
            "Either --bpe-model or --tokens must be provided and point to a valid file."
        )

    logging.info(params)

    # Create model
    logging.info("About to create model")
    model = get_model(params)

    # Load checkpoint
    load_model_checkpoint(params, model, strict=False, device=torch.device("cpu"))
    # Move to CPU and eval
    model.to("cpu")
    model.eval()

    # Route to the appropriate export function
    if params.export_type == "torch":
        export_torch(params, model)
    elif params.export_type == "onnx":
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
