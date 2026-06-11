import collections
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter


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


class MetricsTracker(collections.defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
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

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


def setup_logger(
    log_filename: Path, log_level: str = "info", use_console: bool = True
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
