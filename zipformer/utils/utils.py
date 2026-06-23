# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                2021-2026  Xiaomi Corporation (authors: Fangjun Kuang, Wei Kang)
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
import json
import logging
import math
import pathlib
import random
import warnings
import re

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar, Union
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

import kaldialign
import torch
from torch import distributed as dist
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl
from packaging import version


LOG_EPS = math.log(1e-10)
Pathlike = Union[str, Path]
TORCH_VERSION = version.parse(torch.__version__)

Symbol = TypeVar("Symbol")


# Disable __repr__ otherwise it could freeze e.g. Jupyter.
@dataclass(repr=False)
class SymbolTable(Generic[Symbol]):
    """SymbolTable that maps symbol IDs, found on the FSA arcs to
    actual objects. These objects can be arbitrary Python objects
    that can serve as keys in a dictionary (i.e. they need to be
    hashable and immutable).

    The SymbolTable can only be read to/written from disk if the
    symbols are strings.
    """

    _id2sym: Dict[int, Symbol] = field(default_factory=dict)
    """Map an integer to a symbol.
    """

    _sym2id: Dict[Symbol, int] = field(default_factory=dict)
    """Map a symbol to an integer.
    """

    _next_available_id: int = 1
    """A helper internal field that helps adding new symbols
    to the table efficiently.
    """

    eps: Symbol = "<eps>"
    """Null symbol, always mapped to index 0.
    """

    def __post_init__(self):
        for idx, sym in self._id2sym.items():
            assert self._sym2id[sym] == idx
            assert idx >= 0

        for sym, idx in self._sym2id.items():
            assert idx >= 0
            assert self._id2sym[idx] == sym

        if 0 not in self._id2sym:
            self._id2sym[0] = self.eps
            self._sym2id[self.eps] = 0
        else:
            assert self._id2sym[0] == self.eps
            assert self._sym2id[self.eps] == 0

        self._next_available_id = max(self._id2sym) + 1

    @staticmethod
    def from_str(s: str) -> "SymbolTable":
        """Build a symbol table from a string.

        The string consists of lines. Every line has two fields separated
        by space(s), tab(s) or both. The first field is the symbol and the
        second the integer id of the symbol.

        Args:
          s:
            The input string with the format described above.
        Returns:
          An instance of :class:`SymbolTable`.
        """
        id2sym: Dict[int, str] = dict()
        sym2id: Dict[str, int] = dict()

        for line in s.split("\n"):
            fields = line.split()
            if len(fields) == 0:
                continue  # skip empty lines
            assert len(fields) == 2, (
                f"Expect a line with 2 fields. Given: {len(fields)}"
            )
            sym, idx = fields[0], int(fields[1])
            assert sym not in sym2id, f"Duplicated symbol {sym}"
            assert idx not in id2sym, f"Duplicated id {idx}"
            id2sym[idx] = sym
            sym2id[sym] = idx

        eps = id2sym.get(0, "<eps>")

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=eps)

    @staticmethod
    def from_file(filename: str) -> "SymbolTable":
        """Build a symbol table from file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.

        Returns:
          An instance of :class:`SymbolTable`.

        """
        with open(filename, "r", encoding="utf-8") as f:
            return SymbolTable.from_str(f.read().strip())

    def to_str(self) -> str:
        """
        Returns:
          Return a string representation of this object. You can pass
          it to the method ``from_str`` to recreate an identical object.
        """
        s = ""
        for idx, symbol in sorted(self._id2sym.items()):
            s += f"{symbol} {idx}\n"
        return s

    def to_file(self, filename: str):
        """Serialize the SymbolTable to a file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.
        """
        with open(filename, "w") as f:
            for idx, symbol in sorted(self._id2sym.items()):
                print(symbol, idx, file=f)

    def add(self, symbol: Symbol, index: Optional[int] = None) -> int:
        """Add a new symbol to the SymbolTable.

        Args:
            symbol:
                The symbol to be added.
            index:
                Optional int id to which the symbol should be assigned.
                If it is not available, a ValueError will be raised.

        Returns:
            The int id to which the symbol has been assigned.
        """
        # Already in the table? Return its ID.
        if symbol in self._sym2id:
            return self._sym2id[symbol]
        # Specific ID not provided - use next available.
        if index is None:
            index = self._next_available_id
        # Specific ID provided but not available.
        if index in self._id2sym:
            raise ValueError(
                f"Cannot assign id '{index}' to '{symbol}' - "
                f"already occupied by {self._id2sym[index]}"
            )
        self._sym2id[symbol] = index
        self._id2sym[index] = symbol

        # Update next available ID if needed
        if self._next_available_id <= index:
            self._next_available_id = index + 1

        return index

    def get(self, k: Union[int, Symbol]) -> Union[Symbol, int]:
        """Get a symbol for an id or get an id for a symbol

        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.

        Returns:
          An id or a symbol depending on the given `k`.
        """
        if isinstance(k, int):
            return self._id2sym[k]
        else:
            return self._sym2id[k]

    def merge(self, other: "SymbolTable") -> "SymbolTable":
        """Create a union of two SymbolTables.
        Raises an AssertionError if the same IDs are occupied by
        different symbols.

        Args:
            other:
                A symbol table to merge with ``self``.

        Returns:
            A new symbol table.
        """
        self._check_compatible(other)

        id2sym = {**self._id2sym, **other._id2sym}
        sym2id = {**self._sym2id, **other._sym2id}

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=self.eps)

    def _check_compatible(self, other: "SymbolTable") -> None:
        # Epsilon compatibility
        assert self.eps == other.eps, (
            f"Mismatched epsilon symbol: {self.eps} != {other.eps}"
        )
        # IDs compatibility
        common_ids = set(self._id2sym).intersection(other._id2sym)
        for idx in common_ids:
            assert self[idx] == other[idx], (
                f"ID conflict for id: {idx}, "
                f'self[idx] = "{self[idx]}", '
                f'other[idx] = "{other[idx]}"'
            )
        # Symbols compatibility
        common_symbols = set(self._sym2id).intersection(other._sym2id)
        for sym in common_symbols:
            assert self[sym] == other[sym], (
                f"ID conflict for id: {sym}, "
                f'self[sym] = "{self[sym]}", '
                f'other[sym] = "{other[sym]}"'
            )

    def __getitem__(self, item: Union[int, Symbol]) -> Union[Symbol, int]:
        return self.get(item)

    def __contains__(self, item: Union[int, Symbol]) -> bool:
        if isinstance(item, int):
            return item in self._id2sym
        else:
            return item in self._sym2id

    def __len__(self) -> int:
        return len(self._id2sym)

    def __eq__(self, other: "SymbolTable") -> bool:
        if len(self) != len(other):
            return False

        for s in self.symbols:
            if self[s] != other[s]:
                return False

        return True

    @property
    def ids(self) -> List[int]:
        """Returns a list of integer IDs corresponding to the symbols."""
        ans = list(self._id2sym.keys())
        ans.sort()
        return ans

    @property
    def symbols(self) -> List[Symbol]:
        """Returns a list of symbols (e.g., strings) corresponding to
        the integer IDs.
        """
        ans = list(self._sym2id.keys())
        ans.sort()
        return ans


def num_tokens(
    token_table: SymbolTable, disambig_pattern: str = re.compile(r"^#\d+$")
) -> int:
    """Return the number of tokens excluding those from
    disambiguation symbols.

    Caution:
      0 is not a token ID so it is excluded from the return value.
    """
    symbols = token_table.symbols
    ans = []
    for s in symbols:
        if not disambig_pattern.match(s):
            ans.append(token_table[s])
    num_tokens = len(ans)
    if 0 in ans:
        num_tokens -= 1
    return num_tokens


def token_ids_to_text(token_ids: List[int], token_table: SymbolTable) -> str:
    """Convert token IDs to text using a SymbolTable.

    Supports byte-level BPE tokens in the format <0xNN>.
    """
    text = b""
    for i in token_ids:
        token = token_table[i]
        if len(token) >= 4 and token[:3] == "<0x" and token[-1] == ">":
            byte_val = int(token[1:-1], base=16)
            text += byte_val.to_bytes(1, byteorder="little")
        else:
            text += token.encode(encoding="utf-8")
    return text.decode(encoding="utf-8").replace("▁", " ").strip()


def remove_punctuation(s: str) -> str:
    return re.sub(r"[,\.?!\"，。？！“”：:、<>《》\[\]{}【】;；]", "", s)


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


def get_parameter_groups_with_lrs(
    model: torch.nn.Module,
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
