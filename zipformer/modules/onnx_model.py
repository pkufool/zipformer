#!/usr/bin/env python3
#
# Copyright 2021-2026 Xiaomi Corporation (Author: Wei Kang,
#                                                 Fangjun Kuang)
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

import argparse
import logging

import torch
import numpy as np

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


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit-filename",
        required=True,
        type=str,
        help="Path to the torchscript model",
    )

    parser.add_argument(
        "--onnx-encoder-filename",
        required=True,
        type=str,
        help="Path to the onnx encoder model",
    )

    parser.add_argument(
        "--onnx-decoder-filename",
        required=True,
        type=str,
        help="Path to the onnx decoder model",
    )

    parser.add_argument(
        "--onnx-joiner-filename",
        required=True,
        type=str,
        help="Path to the onnx joiner model",
    )

    return parser

def test_encoder(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxTransducerModel,
):
    C = 80
    for i in range(3):
        N = torch.randint(low=1, high=20, size=(1,)).item()
        T = torch.randint(low=30, high=50, size=(1,)).item()
        logging.info(f"test_encoder: iter {i}, N={N}, T={T}")

        x = torch.rand(N, T, C)
        x_lens = torch.randint(low=30, high=T + 1, size=(N,))
        x_lens[0] = T

        torch_encoder_out, torch_encoder_out_lens = torch_model.encoder(x, x_lens)
        torch_encoder_out = torch_model.joiner.encoder_proj(torch_encoder_out)

        onnx_encoder_out, onnx_encoder_out_lens = onnx_model.run_encoder(x, x_lens)

        assert torch.allclose(torch_encoder_out, onnx_encoder_out, atol=1e-05), (
            (torch_encoder_out - onnx_encoder_out).abs().max()
        )


def test_decoder(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxTransducerModel,
):
    context_size = onnx_model.context_size
    vocab_size = onnx_model.vocab_size
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_decoder: iter {i}, N={N}")
        x = torch.randint(
            low=1,
            high=vocab_size,
            size=(N, context_size),
            dtype=torch.int64,
        )
        torch_decoder_out = torch_model.decoder(x, need_pad=torch.tensor([False]))
        torch_decoder_out = torch_model.joiner.decoder_proj(torch_decoder_out)
        torch_decoder_out = torch_decoder_out.squeeze(1)

        onnx_decoder_out = onnx_model.run_decoder(x)
        assert torch.allclose(torch_decoder_out, onnx_decoder_out, atol=1e-4), (
            (torch_decoder_out - onnx_decoder_out).abs().max()
        )


def test_joiner(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxTransducerModel,
):
    encoder_dim = torch_model.joiner.encoder_proj.weight.shape[1]
    decoder_dim = torch_model.joiner.decoder_proj.weight.shape[1]
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_joiner: iter {i}, N={N}")
        encoder_out = torch.rand(N, encoder_dim)
        decoder_out = torch.rand(N, decoder_dim)

        projected_encoder_out = torch_model.joiner.encoder_proj(encoder_out)
        projected_decoder_out = torch_model.joiner.decoder_proj(decoder_out)

        torch_joiner_out = torch_model.joiner(encoder_out, decoder_out)
        onnx_joiner_out = onnx_model.run_joiner(
            projected_encoder_out, projected_decoder_out
        )

        assert torch.allclose(torch_joiner_out, onnx_joiner_out, atol=1e-4), (
            (torch_joiner_out - onnx_joiner_out).abs().max()
        )


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    torch_model = torch.jit.load(args.jit_filename)

    onnx_model = OnnxTransducerModel(
        encoder_model_filename=args.onnx_encoder_filename,
        decoder_model_filename=args.onnx_decoder_filename,
        joiner_model_filename=args.onnx_joiner_filename,
    )

    logging.info("Test encoder")
    test_encoder(torch_model, onnx_model)

    logging.info("Test decoder")
    test_decoder(torch_model, onnx_model)

    logging.info("Test joiner")
    test_joiner(torch_model, onnx_model)
    logging.info("Finished checking ONNX models")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # See https://github.com/pytorch/pytorch/issues/38342
    # and https://github.com/pytorch/pytorch/issues/33354
    #
    # If we don't do this, the delay increases whenever there is
    # a new request that changes the actual batch size.
    # If you use `py-spy dump --pid <server-pid> --native`, you will
    # see a lot of time is spent in re-compiling the torch script model.
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)
    torch.manual_seed(20220727)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
