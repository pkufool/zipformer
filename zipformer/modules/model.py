# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
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

from typing import Optional, Tuple, Union, List

import k2
import torch
import torch.nn as nn
from lhotse.dataset import SpecAugment
from zipformer.modules.scaling import ScaledLinear
from zipformer.utils.utils import add_sos, make_pad_mask, time_warp, torch_autocast, pad_sequences
from .decoder import Decoder
from .joiner import Joiner
from .attention_decoder import AttentionDecoderModel
from .subsampling import Conv2dSubsampling
from .zipformer import Zipformer
from .scaling import ScheduledFloat, FloatLike


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


class AsrModel(nn.Module):
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

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

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
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.encoder_out_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
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
        cr_loss = nn.functional.kl_div(
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
        sos_y_padded, _ = pad_sequences(y, padding_value=blank_id, sos_id=blank_id, device=encoder_out.device)
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
            targets, target_length = pad_sequences(y, padding_value=0, device=encoder_out.device)
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
