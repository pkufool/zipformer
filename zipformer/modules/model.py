# Copyright    2021-2026  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import Optional, Tuple, Union, List

import torch
from lhotse.dataset import SpecAugment
from zipformer.utils import (
    add_sos,
    make_pad_mask,
    time_warp,
    torch_autocast,
    pad_sequences,
)
from .attention_decoder import AttentionDecoderModel
from .subsampling import Conv2dSubsampling
from .zipformer import Zipformer
from .scaling import ScheduledFloat, FloatLike, ScaledLinear, Balancer

import numpy as np

try:
    import k2
except ImportError:
    k2 = None


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


class Decoder(torch.nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        # the balancers are to avoid any drift in the magnitude of the
        # embeddings, which would interact badly with parameter averaging.
        self.balancer = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size

        if context_size > 1:
            self.conv = torch.nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  # group size == 4
                bias=False,
            )
            self.balancer2 = Balancer(
                decoder_dim,
                channel_dim=-1,
                min_positive=0.0,
                max_positive=1.0,
                min_abs=0.5,
                max_abs=1.0,
                prob=0.05,
            )
        else:
            # To avoid `RuntimeError: Module 'Decoder' has no attribute 'conv'`
            # when inference with torch.jit.script and context_size == 1
            self.conv = torch.nn.Identity()
            self.balancer2 = torch.nn.Identity()

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        embedding_out = self.balancer(embedding_out)

        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = torch.nn.functional.pad(
                    embedding_out, pad=(self.context_size - 1, 0)
                )
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
            embedding_out = torch.nn.functional.relu(embedding_out)
            embedding_out = self.balancer2(embedding_out)

        return embedding_out


class Joiner(torch.nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim, initial_scale=0.25)
        self.output_linear = torch.nn.Linear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim, (
            encoder_out.shape,
            decoder_out.shape,
        )

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit


class AsrModel(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = 80,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
        use_ctc: bool = False,
        blank_id: int = 0,
        vocab_size: int = 500,
        use_transducer: bool = True,
        decoder_dim: int = 512,
        context_size: int = 2,
        joiner_dim: int = 512,
        use_attention_decoder: bool = False,
        attention_decoder_dim: int = 512,
        attention_decoder_num_layers: int = 2,
        attention_decoder_attention_dim: int = 512,
        attention_decoder_num_heads: int = 8,
        attention_decoder_feedforward_dim: int = 2048,
        sos_id: int = 1,
        eos_id: int = 2,
        ignore_id: int = -100,
        label_smoothing: float = 0.0,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
          use_attention_decoder:
            Whether use attention-decoder head. Default: False.
        """
        super().__init__()

        assert use_transducer or use_ctc, (
            f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"
        )

        self.blank_id = blank_id
        self.vocab_size = vocab_size

        # encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7) // 2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7) // 2
        #   (2) embedding: num_features -> encoder_dims
        # In the normal configuration, we will downsample once more at the end
        # by a factor of 2, and most of the encoder stacks will run at a lower
        # sampling rate.
        self.encoder_embed = Conv2dSubsampling(
            in_channels=feature_dim,
            out_channels=_to_int_tuple(encoder_dim)[0],
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        )

        self.encoder = Zipformer(
            output_downsampling_factor=2,
            downsampling_factor=_to_int_tuple(downsampling_factor),
            num_encoder_layers=_to_int_tuple(num_encoder_layers),
            encoder_dim=_to_int_tuple(encoder_dim),
            encoder_unmasked_dim=_to_int_tuple(encoder_unmasked_dim),
            query_head_dim=_to_int_tuple(query_head_dim),
            pos_head_dim=_to_int_tuple(pos_head_dim),
            value_head_dim=_to_int_tuple(value_head_dim),
            pos_dim=pos_dim,
            num_heads=_to_int_tuple(num_heads),
            feedforward_dim=_to_int_tuple(feedforward_dim),
            cnn_module_kernel=_to_int_tuple(cnn_module_kernel),
            dropout=dropout,
            warmup_batches=warmup_batches,
            causal=causal,
            chunk_size=_to_int_tuple(chunk_size),
            left_context_frames=_to_int_tuple(left_context_frames),
        )

        self.use_transducer = use_transducer
        self.encoder_out_dim = max(_to_int_tuple(encoder_dim))
        if use_transducer:
            self.decoder = Decoder(
                vocab_size=vocab_size,
                decoder_dim=decoder_dim,
                blank_id=blank_id,
                context_size=context_size,
            )
            self.joiner = Joiner(
                encoder_dim=self.encoder_out_dim,
                decoder_dim=decoder_dim,
                joiner_dim=joiner_dim,
                vocab_size=vocab_size,
            )
            self.simple_am_proj = ScaledLinear(
                self.encoder_out_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            self.decoder = None
            self.joiner = None

        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_output = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(self.encoder_out_dim, vocab_size),
                torch.nn.LogSoftmax(dim=-1),
            )
        else:
            self.ctc_output = None

        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.attention_decoder = AttentionDecoderModel(
                vocab_size=vocab_size,
                decoder_dim=attention_decoder_dim,
                num_decoder_layers=attention_decoder_num_layers,
                attention_dim=attention_decoder_attention_dim,
                num_heads=attention_decoder_num_heads,
                feedforward_dim=attention_decoder_feedforward_dim,
                memory_dim=self.encoder_out_dim,
                sos_id=sos_id,
                eos_id=eos_id,
                ignore_id=ignore_id,
                label_smoothing=label_smoothing,
            )
        else:
            self.attention_decoder = None

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction=reduction,
        )
        return ctc_loss

    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        ctc_output = self.ctc_output(encoder_out)  # (2 * N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction=reduction,
        )

        # Compute consistency regularization loss
        batch_size = ctc_output.shape[0]
        assert batch_size % 2 == 0, batch_size
        # exchange: [x1, x2] -> [x2, x1]
        exchanged_targets = torch.roll(ctc_output.detach(), batch_size // 2, dims=0)
        cr_loss = torch.nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0)

        if reduction == "sum":
            cr_loss = cr_loss.sum()
        elif reduction == "mean":
            cr_loss = cr_loss.mean()

        return ctc_loss, cr_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: List[List[int]],
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A list of token id list. It contains labels of each utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network) part.
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network) part.
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.blank_id
        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded, _ = pad_sequences(
            y, padding_value=blank_id, sos_id=blank_id, device=encoder_out.device
        )
        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded, y_lens = pad_sequences(y, padding_value=0, device=encoder_out.device)

        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch_autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction=reduction,
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]
        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch_autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction=reduction,
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[List[int]],
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A list of token id list. It contains labels of each utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          use_cr_ctc:
            Whether use consistency-regularized CTC.
          use_spec_aug:
            Whether apply spec-augment manually, used only if use_cr_ctc is True.
          spec_augment:
            The SpecAugment instance that returns time masks,
            used only if use_cr_ctc is True.
          supervision_segments:
            An int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features``.
            Used only if use_cr_ctc is True.
          time_warp_factor:
            Parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
            Used only if use_cr_ctc is True.

        Returns:
          Return the transducer losses, CTC loss, AED loss,
          and consistency-regularization loss in form of
          (simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        assert x.size(0) == x_lens.size(0) == len(y), (x.shape, x_lens.shape, len(y))

        if use_cr_ctc:
            assert self.use_ctc
            if use_spec_aug:
                assert spec_augment is not None and spec_augment.time_warp_factor < 1
                # Apply time warping before input duplicating
                assert supervision_segments is not None
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )
                # Independently apply frequency masking and time masking to the two copies
                x = spec_augment(x.repeat(2, 1, 1))
            else:
                x = x.repeat(2, 1, 1)
            x_lens = x_lens.repeat(2)
            y += y

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
                reduction=reduction,
            )
            if use_cr_ctc:
                simple_loss = simple_loss * 0.5
                pruned_loss = pruned_loss * 0.5
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets, target_length = pad_sequences(
                y, padding_value=0, device=encoder_out.device
            )
            if not use_cr_ctc:
                ctc_loss = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=target_length,
                    reduction=reduction,
                )
                cr_loss = torch.empty(0)
            else:
                ctc_loss, cr_loss = self.forward_cr_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=target_length,
                    reduction=reduction,
                )
                ctc_loss = ctc_loss * 0.5
                cr_loss = cr_loss * 0.5
        else:
            ctc_loss = torch.empty(0)
            cr_loss = torch.empty(0)

        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y,
                reduction=reduction,
            )
            if use_cr_ctc:
                attention_decoder_loss = attention_decoder_loss * 0.5
        else:
            attention_decoder_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss


# The following are wrapper classes for JIT and onnx export.
class EncoderWrapper(torch.nn.Module):
    """A wrapper for encoder and encoder_embed (non-streaming JIT export)."""

    def __init__(
        self, encoder: torch.nn.Module, encoder_embed: torch.nn.Module
    ) -> None:
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


class StreamingEncoderWrapper(torch.nn.Module):
    """A wrapper for encoder and encoder_embed (streaming JIT export)."""

    def __init__(
        self, encoder: torch.nn.Module, encoder_embed: torch.nn.Module
    ) -> None:
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


class OnnxEncoderWrapper(torch.nn.Module):
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


class OnnxDecoderWrapper(torch.nn.Module):
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


class OnnxJoinerWrapper(torch.nn.Module):
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


class OnnxCtcWrapper(torch.nn.Module):
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


class OnnxStreamingEncoderWrapper(torch.nn.Module):
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


class OnnxStreamingCtcWrapper(torch.nn.Module):
    """A wrapper for Zipformer and the ctc_head (streaming)."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        encoder_embed: torch.nn.Module,
        ctc_output: torch.nn.Module,
    ):
        """
        Args:
          encoder:
            A Zipformer encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
          ctc_output:
            The ctc head.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.ctc_output = ctc_output
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3

    def forward(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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

        # processed_mask is used to mask out initial states
        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]  # (batch,)
        # (batch, left_context_size)
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        # Update processed lengths
        new_processed_lens = processed_lens + x_lens
        # (batch, left_context_size + chunk_size)
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
        # Now encoder_out is of shape (N, T, ctc_output_dim)

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
        """
        Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        states[-2] is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        states[-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
        """
        states = self.encoder.get_init_states(batch_size, device)

        embed_states = self.encoder_embed.get_init_states(batch_size, device)

        states.append(embed_states)

        processed_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        states.append(processed_lens)

        return states


# The following classes are used for ONNX inference.
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


# The following code is used to check the correctness of the exported ONNX models.
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit-model",
        required=True,
        type=str,
        help="Path to the torchscript model",
    )

    parser.add_argument(
        "--onnx-encoder",
        required=True,
        type=str,
        help="Path to the onnx encoder model",
    )

    parser.add_argument(
        "--onnx-decoder",
        required=True,
        type=str,
        help="Path to the onnx decoder model",
    )

    parser.add_argument(
        "--onnx-joiner",
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

    torch_model = torch.jit.load(args.jit_model)

    onnx_model = OnnxTransducerModel(
        encoder_model_filename=args.onnx_encoder,
        decoder_model_filename=args.onnx_decoder,
        joiner_model_filename=args.onnx_joiner,
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
