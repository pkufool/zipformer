# Copyright    2021-2026  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                       Wei Kang
#                                                       Xiaoyu Yang)
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

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as spm
import torch
from torch import nn
from .context_graph import ContextGraph
from .lm import ContextState, NgramLm, NgramLmStateCost, LmScorer
from zipformer.utils.utils import (
    DecodingResults,
    KeywordResult,
    add_eos,
    add_sos,
)
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Union
from stream import DecodeStream


@dataclass
class KeywordResult:
    timestamps: List[int]
    hyps: List[int]
    phrase: str

@dataclass
class ASRResults:
    timestamps: List[List[int]]
    hyps: List[List[int]]
    scores: Optional[List[List[float]]] = None


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    ac_probs: Optional[List[float]] = None

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # N-gram LM state
    state_cost: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


# Transducer decoding related classes and functions.
def _greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], ASRResults]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      blank_penalty:
        The score used to penalize blank probability.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a ASRResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return ASRResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )


def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    max_sym_per_frame: int = 1,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], ASRResults]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
      blank_penalty:
        The score used to penalize blank probability.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a ASRResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3, encoder_out.shape

    if max_sym_per_frame == 1:
        return _greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            blank_penalty=blank_penalty,
            return_timestamps=return_timestamps,
        )
    else:
        warnings.warn(
            "max_sym_per_frame > 1 is not supported in batch mode. "
            "Falling back to greedy search for a single utterance."
        )
        # support only batch_size == 1 for now
        assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id)
    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], device=device, dtype=torch.int64
    ).reshape(1, context_size)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # timestamp[i] is the frame index after subsampling
    # on which hyp[i] is decoded
    timestamp = []

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000
    # symbols per frame
    sym_per_frame = 0
    # symbols per utterance decoded so far
    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        # fmt: on
        # logits is (1, 1, 1, vocab_size)
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )

        if blank_penalty != 0:
            logits[:, :, :, 0] -= blank_penalty

        y = logits.argmax().item()
        if y not in (blank_id, unk_id):
            hyp.append(y)
            timestamp.append(t)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)
            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
    hyp = hyp[context_size:]  # remove blanks

    if not return_timestamps:
        return hyp
    else:
        return ASRResults(
            hyps=[hyp],
            timestamps=[timestamp],
        )

class HypsShape:
    """A lightweight replacement for k2.RaggedShape storing row_splits and row_ids."""

    def __init__(self, row_splits: torch.Tensor):
        self._row_splits = row_splits
        self._row_ids = None

    def row_splits(self) -> torch.Tensor:
        return self._row_splits

    def row_ids(self) -> torch.Tensor:
        if self._row_ids is None:
            row_splits = self._row_splits
            num_utt = row_splits.size(0) - 1
            # build row_ids: for utterance i, indices [row_splits[i], row_splits[i+1>) map to i
            total = row_splits[-1].item()
            row_ids = torch.zeros(total, dtype=torch.int32)
            for i in range(num_utt):
                start = row_splits[i].item()
                end = row_splits[i + 1].item()
                row_ids[start:end] = i
            self._row_ids = row_ids.to(self._row_splits.device)
        return self._row_ids

    def to(self, device) -> "HypsShape":
        self._row_splits = self._row_splits.to(device)
        if self._row_ids is not None:
            self._row_ids = self._row_ids.to(device)
        return self


def get_hyps_shape(hyps: List[HypothesisList]) -> HypsShape:
    """Return a ragged shape with axes [utt][num_hyps].

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    return HypsShape(row_splits=row_splits)


def _per_utterance_topk(
    flat_tensor: torch.Tensor,
    row_splits: torch.Tensor,
    k: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Perform topk on each utterance's slice of a flat tensor.

    Args:
      flat_tensor: A 1-D tensor.
      row_splits: row_splits[i] and row_splits[i+1] define the range for utterance i.
      k: Number of top elements to return per utterance.

    Returns:
      A list of (values, indices) tuples, one per utterance.
    """
    results = []
    num_utt = row_splits.size(0) - 1
    for i in range(num_utt):
        start = row_splits[i].item()
        end = row_splits[i + 1].item()
        segment = flat_tensor[start:end]
        values, indices = segment.topk(min(k, segment.size(0)))
        results.append((values, indices))
    return results


def _lm_scoring_pass(
    A: List[List[Hypothesis]],
    batch_size: int,
    per_utt_topk: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    LM: LmScorer,
    blank_id: int,
    unk_id: int,
    context_size: int,
    device: torch.device,
    sos_id: int,
) -> Tuple[List, List, List, int]:
    """First pass of two-pass LM scoring: collect non-blank tokens and score them.

    Returns (scores, lm_states, token_list, count_offset).
    Pass 1: iterate topk results, collect non-blank tokens for LM scoring.
    Pass 2 is done by the caller using the returned scores/lm_states.
    """
    token_list = []
    hs = []
    cs = []
    for i in range(batch_size):
        topk_log_probs, topk_indexes = per_utt_topk[i]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
            topk_token_indexes = (topk_indexes % vocab_size).tolist()
        for k in range(len(topk_hyp_indexes)):
            hyp_idx = topk_hyp_indexes[k]
            hyp = A[i][hyp_idx]

            new_token = topk_token_indexes[k]
            if new_token not in (blank_id, unk_id):
                if LM.lm_type == "rnn":
                    token_list.append([new_token])
                    hs.append(hyp.state[0])
                    cs.append(hyp.state[1])
                else:
                    token_list.append(
                        [sos_id] + hyp.ys[context_size:] + [new_token]
                    )

    scores = None
    lm_states = None
    if len(token_list) != 0:
        x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
        if LM.lm_type == "rnn":
            tokens_to_score = (
                torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
            )
            hs = torch.cat(hs, dim=1).to(device)
            cs = torch.cat(cs, dim=1).to(device)
            state = (hs, cs)
        else:
            tokens_list = [torch.tensor(tokens) for tokens in token_list]
            tokens_to_score = (
                torch.nn.utils.rnn.pad_sequence(
                    tokens_list, batch_first=True, padding_value=0.0
                )
                .to(device)
                .to(torch.int64)
            )
            state = None

        scores, lm_states = LM.score_token(tokens_to_score, x_lens, state)

    return scores, lm_states


def _post_hoc_lm_rescoring(
    B: List[HypothesisList],
    LM: LmScorer,
    context_size: int,
    blank_id: int,
    device: torch.device,
    am_scores: torch.Tensor,
    am_row_splits: torch.Tensor,
    candidate_seqs: List[List[int]],
    lm_scale_list: List[float],
    packed_encoder_out,
    LODR_lm: Optional[NgramLm] = None,
    sp: Optional[spm.SentencePieceProcessor] = None,
) -> Dict[str, List[List[int]]]:
    """Post-hoc LM rescoring for modes 2 (LM rescore) and 3 (LM rescore + LODR)."""
    sentence_token_lengths = torch.tensor(
        [len(s) for s in candidate_seqs], dtype=torch.int64
    )
    seqs_with_sos = add_sos(candidate_seqs, sos_id=1)
    seqs_with_eos = add_eos(candidate_seqs, eos_id=1)
    sentence_token_lengths += 1

    x = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in seqs_with_sos],
        batch_first=True,
        padding_value=blank_id,
    )
    y = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in seqs_with_eos],
        batch_first=True,
        padding_value=blank_id,
    )
    x = x.to(device).to(torch.int64)
    y = y.to(device).to(torch.int64)
    sentence_token_lengths = sentence_token_lengths.to(device).to(torch.int64)

    lm_scores = LM.lm(x=x, y=y, lengths=sentence_token_lengths)
    assert lm_scores.ndim == 2
    lm_scores = -1 * lm_scores.sum(dim=1)

    LODR_scores = None
    if LODR_lm is not None and sp is not None:
        LODR_scores = []
        for seq in candidate_seqs:
            tokens = " ".join(sp.id_to_piece(seq))
            LODR_scores.append(LODR_lm.score(tokens))
        LODR_scores = torch.tensor(LODR_scores).to(device) * math.log(10)
        assert lm_scores.shape == LODR_scores.shape

    ans = {}
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    num_utt = am_row_splits.size(0) - 1

    if LODR_scores is not None:
        LODR_scale_list = [0.05 * i for i in range(1, 20)]
        for lm_scale in lm_scale_list:
            for lodr_scale in LODR_scale_list:
                key = f"nnlm_scale_{lm_scale:.2f}_lodr_scale_{lodr_scale:.2f}"
                tot_scores = (
                    am_scores / lm_scale + lm_scores - LODR_scores * lodr_scale
                )
                max_indexes = []
                for i in range(num_utt):
                    start = am_row_splits[i].item()
                    end = am_row_splits[i + 1].item()
                    max_indexes.append(start + tot_scores[start:end].argmax().item())
                unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
                hyps = [unsorted_hyps[idx] for idx in unsorted_indices]
                ans[key] = hyps
    else:
        for lm_scale in lm_scale_list:
            key = f"nnlm_scale_{lm_scale:.2f}"
            tot_scores = am_scores + lm_scores * lm_scale
            max_indexes = []
            for i in range(num_utt):
                start = am_row_splits[i].item()
                end = am_row_splits[i + 1].item()
                max_indexes.append(start + tot_scores[start:end].argmax().item())
            unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
            hyps = [unsorted_hyps[idx] for idx in unsorted_indices]
            ans[key] = hyps

    return ans


def _finalize_context_graph(
    B: List[HypothesisList],
    context_graph: ContextGraph,
) -> List[HypothesisList]:
    """Finalize context_state: add backoff arc score for unmatched contexts."""
    finalized_B = [HypothesisList() for _ in range(len(B))]
    for i, hyps in enumerate(B):
        for hyp in list(hyps):
            context_score, new_context_state = context_graph.finalize(
                hyp.context_state
            )
            finalized_B[i].add(
                Hypothesis(
                    ys=hyp.ys,
                    log_prob=hyp.log_prob + context_score,
                    timestamp=hyp.timestamp,
                    context_state=new_context_state,
                )
            )
    return finalized_B


def modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    context_graph: Optional[ContextGraph] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 0.0,
    LODR_lm: Optional[NgramLm] = None,
    LODR_lm_scale: float = 0.0,
    sp: Optional[spm.SentencePieceProcessor] = None,
    lm_scale_list: Optional[List[float]] = None,
) -> Union[List[List[int]], DecodingResults, Dict]:
    """Unified modified beam search supporting multiple decoding modes.

    Mode is auto-detected from which optional parameters are provided:
      1. Plain beam search: no LM, no ngram_lm
      2. LM rescore: LM + lm_scale_list (no LODR_lm)
      3. LM rescore + LODR: LM + LODR_lm + sp + lm_scale_list
      4. N-gram rescoring: ngram_lm
      5. LODR shallow fusion: LM + LODR_lm (no lm_scale_list)
      6. LM shallow fusion: LM only (no lm_scale_list, no LODR_lm)
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    # --- Detect mode ---
    mode_ngram = ngram_lm is not None
    mode_lm_shallow = LM is not None and lm_scale_list is None and LODR_lm is None
    mode_lodr_shallow = LM is not None and LODR_lm is not None and lm_scale_list is None
    mode_lm_rescore = LM is not None and lm_scale_list is not None and LODR_lm is None
    mode_lm_rescore_lodr = (
        LM is not None and lm_scale_list is not None and LODR_lm is not None
    )

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    sos_id = getattr(LM, "sos_id", 1) if LM is not None else 1

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    # --- Init hypotheses ---
    init_score, init_states = None, None
    if LM is not None and (mode_lm_shallow or mode_lodr_shallow):
        sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
        lens = torch.tensor([1]).to(device)
        init_score, init_states = LM.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        hyp_kwargs = dict(
            ys=[-1] * (context_size - 1) + [blank_id],
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
        )
        if not mode_ngram and not mode_lm_shallow and not mode_lodr_shallow:
            hyp_kwargs["timestamp"] = []
        if context_graph is not None:
            hyp_kwargs["context_state"] = context_graph.root
        if mode_ngram:
            hyp_kwargs["state_cost"] = NgramLmStateCost(ngram_lm)
        if mode_lodr_shallow:
            hyp_kwargs["state"] = init_states
            hyp_kwargs["lm_score"] = init_score.reshape(-1)
            hyp_kwargs["state_cost"] = NgramLmStateCost(LODR_lm)
        if mode_lm_shallow:
            hyp_kwargs["state"] = init_states
            hyp_kwargs["lm_score"] = init_score.reshape(-1)
        B[i].add(Hypothesis(**hyp_kwargs))

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    # --- Main loop ---
    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        # ys_log_probs: base log_prob, optionally with ngram score baked in
        if mode_ngram:
            ys_log_probs = torch.cat(
                [
                    hyp.log_prob.reshape(1, 1) + hyp.state_cost.lm_score * ngram_lm_scale
                    for hyps in A
                    for hyp in hyps
                ]
            )
        else:
            ys_log_probs = torch.cat(
                [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
            )

        # LM scores for shallow fusion mode
        lm_scores_cat = None
        if mode_lm_shallow:
            lm_scores_cat = torch.cat(
                [hyp.lm_score.reshape(1, -1) for hyps in A for hyp in hyps]
            )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids().to(torch.int64).to(current_encoder_out.device),
        )

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )

        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        if mode_lm_shallow or mode_lodr_shallow:
            log_probs = logits.log_softmax(dim=-1)
        else:
            log_probs = (logits / temperature).log_softmax(dim=-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)
        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits() * vocab_size
        per_utt_topk = _per_utterance_topk(log_probs, row_splits, beam)

        # LM scoring pass for shallow fusion modes
        nn_lm_scores = None
        nn_lm_states = None
        if mode_lm_shallow or mode_lodr_shallow:
            nn_lm_scores, nn_lm_states = _lm_scoring_pass(
                A, batch_size, per_utt_topk, vocab_size, LM,
                blank_id, unk_id, context_size, device, sos_id,
            )

        # --- Second pass: build new hypotheses ---
        count = 0  # for LM scoring index
        for i in range(batch_size):
            topk_log_probs, topk_indexes = per_utt_topk[i]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:] if hasattr(hyp, 'timestamp') else []

                hyp_log_prob = topk_log_probs[k]
                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state

                new_state = None
                new_lm_score = None
                new_state_cost = None

                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)

                    if not mode_ngram and not mode_lm_shallow and not mode_lodr_shallow:
                        new_timestamp.append(t)

                    # context graph scoring
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                            _,
                        ) = context_graph.forward_one_step(
                            hyp.context_state, new_token, strict_mode=False
                        )

                    # ngram in-loop scoring
                    if mode_ngram:
                        new_state_cost = hyp.state_cost.forward_one_step(new_token)
                        # keep only AM score by subtracting ngram contribution
                        hyp_log_prob = topk_log_probs[k] - hyp.state_cost.lm_score * ngram_lm_scale
                    elif mode_lodr_shallow:
                        new_state_cost = hyp.state_cost.forward_one_step(new_token)
                        current_ngram_score = new_state_cost.lm_score - hyp.state_cost.lm_score
                        assert current_ngram_score <= 0.0, (
                            new_state_cost.lm_score,
                            hyp.state_cost.lm_score,
                        )
                        hyp_log_prob += (
                            hyp.lm_score[new_token] * LM.lm_scale
                            + LODR_lm_scale * current_ngram_score
                            + context_score
                        )
                        new_lm_score = nn_lm_scores[count]
                        if LM.lm_type == "rnn":
                            new_state = (
                                nn_lm_states[0][:, count, :].unsqueeze(1),
                                nn_lm_states[1][:, count, :].unsqueeze(1),
                            )
                        count += 1
                        context_score = 0  # already added above
                    elif mode_lm_shallow:
                        if lm_scores_cat is not None:
                            hyp_log_prob += hyp.lm_score[new_token] * LM.lm_scale
                        new_lm_score = nn_lm_scores[count]
                        if LM.lm_type == "rnn":
                            new_state = (
                                nn_lm_states[0][:, count, :].unsqueeze(1),
                                nn_lm_states[1][:, count, :].unsqueeze(1),
                            )
                        count += 1
                        if not mode_ngram:
                            new_timestamp.append(t)
                    elif context_graph is not None:
                        pass  # context_score already set
                else:
                    if mode_ngram:
                        new_state_cost = hyp.state_cost
                    elif mode_lodr_shallow:
                        new_state_cost = hyp.state_cost

                if mode_lm_shallow:
                    new_state = new_state if new_state is not None else hyp.state
                    new_lm_score = new_lm_score if new_lm_score is not None else hyp.lm_score

                new_log_prob = hyp_log_prob + context_score

                build_kwargs = dict(ys=new_ys, log_prob=new_log_prob)
                if mode_ngram:
                    build_kwargs["state_cost"] = new_state_cost
                elif mode_lodr_shallow:
                    build_kwargs["state"] = new_state if new_state is not None else hyp.state
                    build_kwargs["lm_score"] = new_lm_score if new_lm_score is not None else hyp.lm_score
                    build_kwargs["state_cost"] = new_state_cost
                    build_kwargs["context_state"] = new_context_state
                elif mode_lm_shallow:
                    build_kwargs["state"] = new_state
                    build_kwargs["lm_score"] = new_lm_score
                    build_kwargs["timestamp"] = new_timestamp
                elif context_graph is not None:
                    build_kwargs["context_state"] = new_context_state
                    build_kwargs["timestamp"] = new_timestamp
                elif not mode_lm_rescore and not mode_lm_rescore_lodr:
                    build_kwargs["timestamp"] = new_timestamp

                new_hyp = Hypothesis(**build_kwargs)
                B[i].add(new_hyp)

    B = B + finalized_B

    # --- Finalization ---
    if context_graph is not None:
        B = _finalize_context_graph(B, context_graph)

    if mode_lm_rescore or mode_lm_rescore_lodr:
        hyps_shape = get_hyps_shape(B)
        am_scores = torch.tensor(
            [hyp.log_prob.item() for b in B for hyp in b]
        ).to(device)
        am_row_splits = hyps_shape.row_splits()
        candidate_seqs = [hyp.ys[context_size:] for b in B for hyp in b]
        return _post_hoc_lm_rescoring(
            B, LM, context_size, blank_id, device,
            am_scores, am_row_splits, candidate_seqs, lm_scale_list,
            packed_encoder_out, LODR_lm=LODR_lm, sp=sp,
        )

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [getattr(h, "timestamp", []) for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return ASRResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )


def beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
    """
    It implements Algorithm 1 in https://arxiv.org/pdf/1211.3711.pdf

    espnet/nets/beam_search_transducer.py#L247 is used as a reference.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      beam:
        Beam size.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.

    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [blank_id] * context_size,
        device=device,
        dtype=torch.int64,
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[-1] * (context_size - 1) + [blank_id], log_prob=0.0, timestamp=[]
        )
    )

    max_sym_per_utt = 20000

    sym_per_utt = 0

    decoder_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        # fmt: on
        A = B
        B = HypothesisList()

        joint_cache: Dict[str, torch.Tensor] = {}

        # TODO(fangjun): Implement prefix search to update the `log_prob`
        # of hypotheses in A

        while True:
            y_star = A.get_most_probable()
            A.remove(y_star)

            cached_key = y_star.key

            if cached_key not in decoder_cache:
                decoder_input = torch.tensor(
                    [y_star.ys[-context_size:]],
                    device=device,
                    dtype=torch.int64,
                ).reshape(1, context_size)

                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
                decoder_cache[cached_key] = decoder_out
            else:
                decoder_out = decoder_cache[cached_key]

            cached_key += f"-t-{t}"
            if cached_key not in joint_cache:
                logits = model.joiner(
                    current_encoder_out,
                    decoder_out.unsqueeze(1),
                    project_input=False,
                )

                if blank_penalty != 0:
                    logits[:, :, :, 0] -= blank_penalty

                # TODO(fangjun): Scale the blank posterior
                log_prob = (logits / temperature).log_softmax(dim=-1)
                # log_prob is (1, 1, 1, vocab_size)
                log_prob = log_prob.squeeze()
                # Now log_prob is (vocab_size,)
                joint_cache[cached_key] = log_prob
            else:
                log_prob = joint_cache[cached_key]

            # First, process the blank symbol
            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob

            # ys[:] returns a copy of ys
            B.add(
                Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    timestamp=y_star.timestamp[:],
                )
            )

            # Second, process other non-blank labels
            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i in (blank_id, unk_id):
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                new_timestamp = y_star.timestamp + [t]
                A.add(
                    Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        timestamp=new_timestamp,
                    )
                )

            # Check whether B contains more than "beam" elements more probable
            # than the most probable in A
            A_most_probable = A.get_most_probable()

            kept_B = B.filter(A_most_probable.log_prob)

            if len(kept_B) >= beam:
                B = kept_B.topk(beam)
                break

        t += 1

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  # [context_size:] to remove blanks

    if not return_timestamps:
        return ys
    else:
        return DecodingResults(hyps=[ys], timestamps=[best_hyp.timestamp])

def ctc_greedy_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    batch = ctc_output.shape[0]
    index = ctc_output.argmax(dim=-1)
    hyps = [
        torch.unique_consecutive(index[i, : encoder_out_lens[i]]) for i in range(batch)
    ]
    return [h[h != blank_id].tolist() for h in hyps]


@dataclass
class Hypothesis:
    ys: List[int] = field(default_factory=list)
    log_prob_blank: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.float32))
    log_prob_non_blank: torch.Tensor = field(
        default_factory=lambda: torch.tensor([float("-inf")], dtype=torch.float32)
    )

    @property
    def tot_score(self) -> torch.Tensor:
        return torch.logaddexp(self.log_prob_non_blank, self.log_prob_blank)

    @property
    def key(self) -> tuple:
        return tuple(self.ys)

    def clone(self) -> "Hypothesis":
        return Hypothesis(
            ys=self.ys,
            log_prob_blank=self.log_prob_blank,
            log_prob_non_blank=self.log_prob_non_blank,
        )


class HypothesisList:
    def __init__(self, data: Optional[Dict[tuple, Hypothesis]] = None) -> None:
        self._data = {} if data is None else data

    def add(self, hyp: Hypothesis) -> None:
        key = hyp.key
        if key in self._data:
            old_hyp = self._data[key]
            torch.logaddexp(old_hyp.log_prob_blank, hyp.log_prob_blank, out=old_hyp.log_prob_blank)
            torch.logaddexp(
                old_hyp.log_prob_non_blank,
                hyp.log_prob_non_blank,
                out=old_hyp.log_prob_non_blank,
            )
        else:
            self._data[key] = hyp

    def get_most_probable(self) -> Hypothesis:
        return max(self._data.values(), key=lambda hyp: hyp.tot_score)

    def topk(self, k: int) -> "HypothesisList":
        hyps = sorted(self._data.items(), key=lambda h: h[1].tot_score, reverse=True)[:k]
        return HypothesisList(dict(hyps))

    def __iter__(self):
        return iter(self._data.values())


def _step_worker(
    log_probs: torch.Tensor,
    indexes: torch.Tensor,
    B: HypothesisList,
    beam: int = 4,
    blank_id: int = 0,
) -> HypothesisList:
    A = list(B)
    B = HypothesisList()
    for hyp in A:
        for k in range(log_probs.size(0)):
            log_prob, index = log_probs[k], indexes[k]
            new_token = index.item()
            new_hyp = hyp.clone()
            if new_token == blank_id:
                new_hyp.log_prob_non_blank = torch.tensor([float("-inf")], dtype=torch.float32)
                new_hyp.log_prob_blank = hyp.tot_score + log_prob
                B.add(new_hyp)
            elif len(hyp.ys) > 0 and hyp.ys[-1] == new_token:
                new_hyp.log_prob_non_blank = hyp.log_prob_non_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor([float("-inf")], dtype=torch.float32)
                B.add(new_hyp)

                new_hyp = hyp.clone()
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor([float("-inf")], dtype=torch.float32)
                B.add(new_hyp)
            else:
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.tot_score + log_prob
                new_hyp.log_prob_blank = torch.tensor([float("-inf")], dtype=torch.float32)
                B.add(new_hyp)
    return B.topk(beam)


def _sequence_worker(
    topk_values: torch.Tensor,
    topk_indexes: torch.Tensor,
    B: HypothesisList,
    encoder_out_lens: int,
    beam: int = 4,
    blank_id: int = 0,
) -> HypothesisList:
    B.add(Hypothesis())
    for j in range(encoder_out_lens):
        B = _step_worker(topk_values[j], topk_indexes[j], B, beam, blank_id)
    return B


def ctc_prefix_beam_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    process_pool: Optional[Pool] = None,
    return_nbest: Optional[bool] = False,
) -> Union[List[List[int]], List[HypothesisList]]:
    batch_size, _, _ = ctc_output.shape
    topk_values, topk_indexes = ctc_output.topk(beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()

    B = [HypothesisList() for _ in range(batch_size)]

    pool = Pool() if process_pool is None else process_pool
    args = [
        (topk_values[i], topk_indexes[i], B[i], encoder_out_lens[i].item(), beam, blank_id)
        for i in range(batch_size)
    ]
    async_results = pool.starmap_async(_sequence_worker, args)
    B = list(async_results.get())
    if process_pool is None:
        pool.close()
        pool.join()

    if return_nbest:
        return B
    return [b.get_most_probable().ys for b in B]

def streaming_greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
    blank_penalty: float = 0.0,
) -> None:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      streams:
        A list of Stream objects.
    """
    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device
    T = encoder_out.size(1)

    decoder_input = torch.tensor(
        [stream.hyp[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    # decoder_out is of shape (N, 1, decoder_out_dim)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    for t in range(T):
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        current_encoder_out = encoder_out[:, t : t + 1, :]  # noqa

        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size,  vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0.0:
            logits[:, 0] -= blank_penalty

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                streams[i].hyp.append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            )
            decoder_out = model.joiner.decoder_proj(decoder_out)


def streaming_modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
    num_active_paths: int = 4,
    blank_penalty: float = 0.0,
) -> None:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The RNN-T model.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      streams:
        A list of stream objects.
      num_active_paths:
        Number of active paths during the beam search.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert len(streams) == encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    batch_size = len(streams)
    T = encoder_out.size(1)

    B = [stream.hyps for stream in streams]

    for t in range(T):
        current_encoder_out = encoder_out[:, t].unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.stack(
            [hyp.log_prob.reshape(1) for hyps in A for hyp in hyps], dim=0
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out, project_input=False)
        # logits is of shape (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0.0:
            logits[:, 0] -= blank_penalty

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        per_utt_topk = _per_utterance_topk(log_probs, row_splits, num_active_paths)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = per_utt_topk[i]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                if new_token != blank_id:
                    new_ys.append(new_token)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                B[i].add(new_hyp)

    for i in range(batch_size):
        streams[i].hyps = B[i]


# transduer keywords decoding.
def keywords_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    keywords_graph: ContextGraph,
    beam: int = 4,
    num_tailing_blanks: int = 0,
    blank_penalty: float = 0,
) -> List[List[KeywordResult]]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      keywords_graph:
        A instance of ContextGraph containing keywords and their configurations.
      beam:
        Number of active paths during the beam search.
      num_tailing_blanks:
        The number of tailing blanks a keyword should be followed, this is for the
        scenario that a keyword will be the prefix of another. In most cases, you
        can just set it to 0.
      blank_penalty:
        The score used to penalize blank probability.
    Returns:
      Return a list of list of KeywordResult.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert keywords_graph is not None

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                context_state=keywords_graph.root,
                timestamp=[],
                ac_probs=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    sorted_ans = [[] for _ in range(N)]
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]

        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        probs = logits.softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs = probs.log()

        probs = probs.reshape(-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        per_utt_topk = _per_utterance_topk(log_probs, row_splits, beam)
        per_utt_probs = _per_utterance_topk(probs, row_splits, beam)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = per_utt_topk[i]
            hyp_probs = per_utt_probs[i][0].tolist()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                new_ac_probs = hyp.ac_probs[:]
                context_score = 0
                new_context_state = hyp.context_state
                new_num_tailing_blanks = hyp.num_tailing_blanks + 1
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    new_ac_probs.append(hyp_probs[topk_indexes[k]])
                    (
                        context_score,
                        new_context_state,
                        _,
                    ) = keywords_graph.forward_one_step(hyp.context_state, new_token)
                    new_num_tailing_blanks = 0
                    if new_context_state.token == -1:  # root
                        new_ys[-context_size:] = [-1] * (context_size - 1) + [blank_id]

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                    ac_probs=new_ac_probs,
                    context_state=new_context_state,
                    num_tailing_blanks=new_num_tailing_blanks,
                )
                B[i].add(new_hyp)

            top_hyp = B[i].get_most_probable(length_norm=True)
            matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
            if matched:
                ac_prob = (
                    sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
                )
            if (
                matched
                and top_hyp.num_tailing_blanks > num_tailing_blanks
                and ac_prob >= matched_state.ac_threshold
            ):
                keyword = KeywordResult(
                    hyps=top_hyp.ys[-matched_state.level :],
                    timestamps=top_hyp.timestamp[-matched_state.level :],
                    phrase=matched_state.phrase,
                )
                sorted_ans[i].append(keyword)
                B[i] = HypothesisList()
                B[i].add(
                    Hypothesis(
                        ys=[-1] * (context_size - 1) + [blank_id],
                        log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                        context_state=keywords_graph.root,
                        timestamp=[],
                        ac_probs=[],
                    )
                )

    B = B + finalized_B

    for i, hyps in enumerate(B):
        top_hyp = hyps.get_most_probable(length_norm=True)
        matched, matched_state = keywords_graph.is_matched(top_hyp.context_state)
        if matched:
            ac_prob = (
                sum(top_hyp.ac_probs[-matched_state.level :]) / matched_state.level
            )
        if matched and ac_prob >= matched_state.ac_threshold:
            keyword = KeywordResult(
                hyps=top_hyp.ys[-matched_state.level :],
                timestamps=top_hyp.timestamp[-matched_state.level :],
                phrase=matched_state.phrase,
            )
            sorted_ans[i].append(keyword)

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans