#!/usr/bin/env python3
# Copyright    2021-2026  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Daniel Povey,
#                                                       Zengwei Yao,
#                                                       Fangjun Kuang)
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
import copy
import logging
import re
import uuid
import warnings

from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp

from atdataset import ATDataloader, Fbank, SpecAugment
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from ssentencepiece import Ssentencepiece

from zipformer.modules.model import AsrModel
from zipformer.modules.optim import Eden, LRScheduler, ScaledAdam

from zipformer.utils import save_checkpoint as save_checkpoint_impl

from zipformer.utils import (
    load_checkpoint,
    remove_checkpoints,
    replace_punctuation_with_space,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
    cleanup_dist,
    fix_random_seed,
    setup_dist,
    get_env_info,
    raise_grad_scale_is_too_small_error,
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[torch.nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying torch.nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="The dimension of input features.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input audio.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )

    parser.add_argument(
        "--attention-decoder-dim",
        type=int,
        default=512,
        help="""Dimension used in the attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-dim",
        type=int,
        default=512,
        help="""Attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-dim",
        type=int,
        default=2048,
        help="""Feedforward dimension used in attention decoder""",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=True,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-cr-ctc",
        type=str2bool,
        default=True,
        help="If True, use consistency-regularized CTC.",
    )


def add_dataloader_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enable-spec-aug",
        type=str2bool,
        default=False,
        help="When enabled, use SpecAugment for training dataset.",
    )

    parser.add_argument(
        "--spec-aug-time-warp-factor",
        type=int,
        default=80,
        help="Used only when --enable-spec-aug is True. "
        "It specifies the factor for time warping in SpecAugment. "
        "Larger values mean more warping. "
        "A value less than 1 means to disable time warp.",
    )

    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=2.5,
        help="When using cr-ctc, we increase the amount of time-masking in SpecAugment.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to use for each data loader.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=200,
        help="Maximum duration (in seconds) of each batch during training. ",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum num of samples of each batch during training. ",
    )

    parser.add_argument(
        "--training-sets",
        nargs="+",
        help="A list of training sets, e.g., "
        "librispeech-train-clean-100 librispeech-train-clean-360 librispeech-train-other-500",
    )

    parser.add_argument(
        "--training-weights",
        type=str,
        help="A comma-separated list of weights for each training sets. ",
    )

    parser.add_argument(
        "--epoch-hours",
        type=float,
        help="Number of hours to process in each epoch.",
    )

    parser.add_argument(
        "--validation-sets",
        nargs="*",
        help="A list of validation sets, e.g., librispeech-dev-clean librispeech-dev-other",
    )

    parser.add_argument(
        "--validation-weights",
        type=str,
        help="A comma-separated list of weights for each validation sets. ",
    )

    parser.add_argument(
        "--use-noise-augment",
        type=str2bool,
        default=True,
        help="Whether to use noise augment for training.",
    )

    parser.add_argument(
        "--noise-list",
        type=str,
        default="data/tars/musan.lst",
        help="The noise list used for exp-augment.",
    )

    parser.add_argument(
        "--use-speed-perturb",
        type=str2bool,
        default=True,
        help="Whether to use speed perturbation for training.",
    )

    parser.add_argument(
        "--use-volume-perturb",
        type=str2bool,
        default=True,
        help="Whether to use volume perturbation for training.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--master-addr",
        type=str,
        help="Master node address for DDP training (used in multi-machine setup).",
    )

    parser.add_argument(
        "--local-rank-start",
        type=int,
        default=0,
        help="""Start rank of processes on the current machine, used in multi-machine
        setup, e.g., 0 for first machine, 8 for second).""",
    )

    parser.add_argument(
        "--local-world-size",
        type=int,
        help="""Number of processes (GPUs) on the current machine, used in 
        multi-machine setup""",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="zh-en-8776",
        help="Name or path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-hours",
        type=float,
        default=50000,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--cr-loss-scale",
        type=float,
        default=0.2,
        help="Scale for consistency-regularization loss.",
    )

    parser.add_argument(
        "--attention-decoder-loss-scale",
        type=float,
        default=0.8,
        help="Scale for attention-decoder loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )

    add_model_arguments(parser)
    add_dataloader_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 200,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "subsampling_factor": 4,  # not passed in, this is fixed.
            # parameters for attention-decoder
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def get_model(params: AttributeDict) -> torch.nn.Module:
    model = AsrModel(
        feature_dim=params.feature_dim,
        downsampling_factor=params.downsampling_factor,
        encoder_dim=params.encoder_dim,
        num_encoder_layers=params.num_encoder_layers,
        encoder_unmasked_dim=params.encoder_unmasked_dim,
        query_head_dim=params.query_head_dim,
        pos_head_dim=params.pos_head_dim,
        value_head_dim=params.value_head_dim,
        num_heads=params.num_heads,
        feedforward_dim=params.feedforward_dim,
        cnn_module_kernel=params.cnn_module_kernel,
        pos_dim=params.pos_dim,
        causal=params.causal,
        chunk_size=params.chunk_size,
        left_context_frames=params.left_context_frames,
        use_ctc=params.use_ctc,
        blank_id=params.blank_id,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        decoder_dim=params.decoder_dim,
        context_size=params.context_size,
        joiner_dim=params.joiner_dim,
        use_attention_decoder=params.use_attention_decoder,
        attention_decoder_dim=params.attention_decoder_dim,
        attention_decoder_num_layers=params.attention_decoder_num_layers,
        attention_decoder_attention_dim=params.attention_decoder_attention_dim,
        attention_decoder_num_heads=params.attention_decoder_num_heads,
        attention_decoder_feedforward_dim=params.attention_decoder_feedforward_dim,
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return model


def get_spec_augment(params: AttributeDict) -> SpecAugment:
    num_frame_masks = int(10 * params.time_mask_ratio)
    max_frames_mask_fraction = 0.15 * params.time_mask_ratio
    logging.info(
        f"num_frame_masks: {num_frame_masks}, "
        f"max_frames_mask_fraction: {max_frames_mask_fraction}"
    )
    spec_augment = SpecAugment(
        time_warp_factor=0,  # Do time warping in model.py
        num_frame_masks=num_frame_masks,  # default: 10
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=max_frames_mask_fraction,  # default: 0.15
    )
    return spec_augment


def load_checkpoint_if_available(
    params: AttributeDict,
    model: torch.nn.Module,
    model_avg: torch.nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch - 1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    model_avg: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    sp: Ssentencepiece,
    batch: dict,
    is_training: bool,
    spec_augment: Optional[SpecAugment] = None,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["feature"].to(device)
    feature_lens = batch["feature_lens"].to(device)
    texts = batch["text"]
    y = sp.encode(texts, out_type=int)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    use_cr_ctc = params.use_cr_ctc
    use_spec_aug = use_cr_ctc and is_training

    if use_spec_aug:
        batch_size = len(texts)
        supervision_segments = torch.stack(
            [
                torch.arange(batch_size, dtype=torch.int64),
                torch.zeros(batch_size, dtype=torch.int64),
                feature_lens.to("cpu"),
            ],
            dim=1,
        )  # shape: (S, 3)
    else:
        supervision_segments = None

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            use_cr_ctc=use_cr_ctc,
            use_spec_aug=use_spec_aug,
            spec_augment=spec_augment,
            supervision_segments=supervision_segments,
            time_warp_factor=params.spec_aug_time_warp_factor,
        )

        loss = 0.0

        if params.use_transducer:
            s = params.simple_loss_scale
            # take down the scale on the simple loss from 1.0 at the start
            # to params.simple_loss scale by warm_step.
            simple_loss_scale = (
                s
                if batch_idx_train >= warm_step
                else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
            )
            pruned_loss_scale = (
                1.0
                if batch_idx_train >= warm_step
                else 0.1 + 0.9 * (batch_idx_train / warm_step)
            )
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

        if params.use_ctc:
            loss += params.ctc_loss_scale * ctc_loss
            if use_cr_ctc:
                loss += params.cr_loss_scale * cr_loss

        if params.use_attention_decoder:
            loss += params.attention_decoder_loss_scale * attention_decoder_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if params.use_ctc:
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
        if params.use_cr_ctc:
            info["cr_loss"] = cr_loss.detach().cpu().item()
    if params.use_attention_decoder:
        info["attn_decoder_loss"] = attention_decoder_loss.detach().cpu().item()
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    sp: Ssentencepiece,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for _, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: Ssentencepiece,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    spec_augment: Optional[SpecAugment] = None,
    model_avg: Optional[torch.nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      spec_augment:
        The SpecAugment instance used only when use_cr_ctc is True.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(
        tqdm(train_dl, total=len(train_dl), dynamic_ncols=True)
    ):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["ids"])

        try:
            with torch.amp.autocast(
                "cuda", enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    spec_augment=spec_augment,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)
            scheduler.step_epoch(
                params.batch_idx_train * params.max_duration * params.world_size / 3600
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if (
            valid_dl is not None
            and batch_idx % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated() // 1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def filter_func(sample: Dict[str, Any], sp: Ssentencepiece, sample_rate: int) -> bool:
    T = sample["audio"].size(1) / sample_rate  # in seconds
    T = int(T * 100)  # in 10 ms units
    T = ((T - 7) // 2 + 1) // 2
    text = sample["text"]
    tokens = sp.encode(text)
    S = len(tokens)
    # filter empty text, too short or too long text, and too long audio
    if S == 0 or T - S <= 5 or T / S > 25:
        return False
    return True


def map_func(sample):
    text = replace_punctuation_with_space(sample["text"])
    # remove extra spaces and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()
    sample["text"] = text
    return sample


def run(local_rank, world_size, args):
    """
    Args:
      local_rank:
        Rank of the process on current node, passed by mp.spawn.
      world_size:
        Total number of GPUs across nodes.
      args:
        Parsed command line arguments.
    """
    global_rank = args.local_rank_start + local_rank
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(
            global_rank, world_size, params.master_port, master_addr=params.master_addr
        )

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and global_rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    logging.info(f"Device: {device}")

    sp = Ssentencepiece(params.bpe_model)

    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos>")
    params.vocab_size = sp.vocab_size()

    if not params.use_transducer:
        if not params.use_attention_decoder:
            params.ctc_loss_scale = 1.0
        else:
            assert params.ctc_loss_scale + params.attention_decoder_loss_scale == 1.0, (
                params.ctc_loss_scale,
                params.attention_decoder_loss_scale,
            )

    if params.use_bf16:
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:
        params.dtype = torch.float16
        params.use_autocast = True
    else:
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if params.use_cr_ctc:
        assert params.use_ctc
        assert not params.enable_spec_aug  # we will do spec_augment in model.py
        spec_augment = get_spec_augment(params)
    else:
        spec_augment = None

    assert params.save_every_n >= params.average_period
    model_avg: Optional[torch.nn.Module] = None
    if global_rank == 0:
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_hours, warmup_start=0.1)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    training_sets = params.training_sets
    training_weights = None
    assert training_sets is not None and len(training_sets) > 0, (
        "training_sets must be provided"
    )
    if params.training_weights is not None:
        training_weights = list(map(float, params.training_weights.split(",")))
        assert len(training_weights) == len(training_sets)

    validation_sets = params.validation_sets
    validation_weights = None
    if params.validation_weights is not None:
        validation_weights = list(map(float, params.validation_weights.split(",")))
        assert validation_sets is not None
        assert len(validation_weights) == len(validation_sets)

    feature_extractor = Fbank(
        sample_rate=params.sample_rate,
        n_mels=params.feature_dim,
    )

    _filter_func = partial(filter_func, sp=sp, sample_rate=params.sample_rate)

    train_dl = ATDataloader(
        datasets=training_sets,
        epoch_hours=params.epoch_hours,
        mux_weights=training_weights,
        max_duration=params.max_duration,
        max_samples=params.max_samples,
        min_length=0.5,
        map_func=map_func,
        filter_func=_filter_func,
        feature_extractor=feature_extractor,
        sample_rate=params.sample_rate,
        use_noise_augment=params.use_noise_augment,
        noise_manifest=params.noise_list,
        use_speed_perturb=params.use_speed_perturb,
        use_volume_perturb=params.use_volume_perturb,
        is_test=False,
        num_workers=params.num_workers,
    )

    valid_dl = None
    if validation_sets is not None and len(validation_sets) > 0:
        valid_dl = ATDataloader(
            manifests=validation_sets,
            mux_weights=validation_weights,
            max_duration=params.max_duration,
            max_samples=params.max_samples,
            feature_extractor=feature_extractor,
            min_length=0.5,
            map_func=map_func,
            filter_func=_filter_func,
            sample_rate=params.sample_rate,
            noise_manifest=None,
            is_test=False,
            num_workers=params.num_workers,
        )

    scaler = GradScaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            spec_augment=spec_augment,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=global_rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=global_rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: Ssentencepiece,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    filename = f"{params.exp_dir}/batch-{uuid.uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    feature = batch["feature"]

    logging.info(f"feature shape: {feature.shape}")

    y = sp.encode(batch["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        if args.local_world_size is None:
            local_world_size = world_size
        else:
            local_world_size = args.local_world_size
        mp.spawn(run, args=(world_size, args), nprocs=local_world_size, join=True)
    else:
        run(local_rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
