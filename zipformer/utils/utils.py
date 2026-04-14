import argparse
import collections
import json
import logging
import os
import pathlib
import random
import warnings
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

import kaldialign
import torch
import torch.distributed as dist
import torch.nn as nn
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl
from packaging import version
from torch.utils.tensorboard import SummaryWriter

import socket
import subprocess
import sys

from torch import distributed as dist

Pathlike = Union[str, Path]
TORCH_VERSION = version.parse(torch.__version__)


@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    if TORCH_VERSION >= version.parse("2.3.0"):
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logger(
    log_filename: Pathlike, log_level: str = "info", use_console: bool = True
) -> None:
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.CRITICAL,
    }.get(log_level, logging.ERROR)

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            if isinstance(v, (pathlib.Path, torch.device, torch.dtype)):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)


def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]], char_level: bool = False
) -> None:
    with open(filename, "w", encoding="utf8") as f:
        for cut_id, ref, hyp in texts:
            if char_level:
                ref = list("".join(ref))
                hyp = list("".join(hyp))
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str, str]],
    enable_log: bool = True,
    compute_CER: bool = False,
    sclite_mode: bool = False,
) -> float:
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"

    if compute_CER:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results[i] = (cut_id, ref, hyp)

    for _, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1

    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} [{tot_errs} / {ref_len}, {ins_errs} ins, {del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    return float(tot_err_rate)


class MetricsTracker(collections.defaultdict):
    def __init__(self):
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def norm_items(self) -> List[Tuple[str, float]]:
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k in ("frames", "utterances"):
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self, tb_writer: SummaryWriter, prefix: str, batch_idx: int
    ) -> None:
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


# `is_module_available` is copied from
# https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`.

    Note: "borrowed" from torchaudio:
    """
    import importlib

    return all(importlib.util.find_spec(m) is not None for m in modules)


def make_pad_mask(
    lengths: torch.Tensor, max_len: int = 0, pad_left: bool = False
) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    if pad_left:
        return expanded_lengths < (max_len - lengths).unsqueeze(1)
    return expanded_lengths >= lengths.unsqueeze(-1)


def get_parameter_groups_with_lrs(
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: List[str] = [],
) -> List[dict]:
    flat_lr_scale = defaultdict(lambda: 1.0)
    for name, m in model.named_modules():
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    lr_to_params = defaultdict(list)
    for name, parameter in model.named_parameters():
        split_name = name.split(".")
        prefix = split_name[0]
        if prefix == "module":
            module_name = split_name[1]
            if module_name in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        elif prefix in freeze_modules:
            logging.info(f"Remove {name} from parameters")
            continue

        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [
            {"named_params": pairs, "lr": lr_val}
            for lr_val, pairs in lr_to_params.items()
        ]
    return [{"params": params, "lr": lr_val} for lr_val, params in lr_to_params.items()]


def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
):
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert len(features.shape) == 3, (
        f"SpecAugment only supports 3D tensors: {features.shape}"
    )
    features = features.clone()
    if supervision_segments is None:
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )
    return features


def get_git_sha1():
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                subprocess.run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + "-dirty" if dirty_commit else git_commit + "-clean"
    except:  # noqa
        return None

    return git_commit


def get_git_date():
    try:
        git_date = (
            subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=local"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_git_branch_name():
    try:
        git_date = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:  # noqa
        return None

    return git_date


def get_env_info() -> Dict[str, Any]:
    """Get the environment information."""
    return {
        "torch-version": str(torch.__version__),
        "torch-cuda-available": torch.cuda.is_available(),
        "torch-cuda-version": torch.version.cuda,
        "python-version": ".".join(sys.version.split(".")[:2]),
        "hostname": socket.gethostname(),
        "IP address": socket.gethostbyname(socket.gethostname()),
    }


def raise_grad_scale_is_too_small_error(cur_grad_scale: float):
    raise RuntimeError(
        f"""
        grad_scale is too small, exiting: {cur_grad_scale}

        ========================= NOTE =========================
        If you see this error, it means that the gradient scale is too small.

        The default base_lr is 0.045 / 0.05 (depends on which recipe you are 
        using), this is an empirical value obtained mostly using 4 * 32GB V100 
        GPUs with a max_duration of approx. 1,000. 
        The proper value of base_lr may vary depending on the number of GPUs 
        and the value of max-duration you are using. 

        To fix this issue, you may need to adjust the value of base_lr accordingly.

        We would suggest you to decrease the value of base_lr by 0.005 (e.g., 
        from 0.045 to 0.04), and try again. If the error still exists, you may 
        repeat the process until base_lr hits 0.02. (Note that this will lead to 
        certain loss of performance, but it should work. You can compensate this by
        increasing the num_epochs.)
        
        If the error still exists, you could try to seek help by raising an issue, 
        with a detailed description of (a) your computational resources, (b) the 
        base_lr and (c) the max_duration you are using, (d) detailed configuration 
        of your model.

        ========================================================
        """
    )


def setup_dist(
    rank=None, world_size=None, master_port=None, use_ddp_launch=False, master_addr=None
):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        local_device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_device_id)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    dist.destroy_process_group()


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def add_sos(seq: List[List[int]], sos_id: int) -> List[List[int]]:
    return [[sos_id] + s for s in seq]


def add_eos(seq: List[List[int]], eos_id: int) -> List[List[int]]:
    return [s + [eos_id] for s in seq]


def pad_sequences(
    seq: List[List[int]],
    padding_value: int,
    sos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length with a specified padding value.
    Optionally, add SOS and EOS tokens.
    Args:
        seq: A list of sequences, where each sequence is a list of integers.
        padding_value: The value to use for padding.
        sos_id: If not None, the ID to use for the start-of-sequence token.
                If None, no SOS token will be added.
        eos_id: If not None, the ID to use for the end-of-sequence token.
                If None, no EOS token will be added.
        device: The device on which to create the output tensor.
                If None, the output tensor will be created on the CPU.

    Returns:
        A tuple of two tensors:
        - A tensor of shape (batch_size, max_len) with the padded sequences.
        - A tensor of shape (batch_size,) with the lengths of each sequence.
    """
    batch_size = len(seq)
    seq_lens = []
    max_len = 0
    for s in seq:
        length = len(s)
        if sos_id is not None:
            length += 1
        if eos_id is not None:
            length += 1
        seq_lens.append(length)
        if length > max_len:
            max_len = length
    out = torch.full(
        (batch_size, max_len),
        fill_value=padding_value,
        dtype=torch.int64,
        device=device,
    )
    for i, s in enumerate(seq):
        if sos_id is not None:
            out[i, 0] = sos_id
        if len(s) > 0:
            out[
                i,
                (1 if sos_id is not None else 0) : (
                    len(s) + (1 if sos_id is not None else 0)
                ),
            ] = torch.tensor(s, dtype=torch.int64, device=device)
        if eos_id is not None:
            out[i, len(s) + (1 if sos_id is not None else 0)] = eos_id
    out_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    return out, out_lens


def tokenize_by_cjk_char(line: str) -> List[str]:
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip())
    words: List[str] = []
    for w in chars:
        if w.strip():
            words.extend(w.split())
    return words
