# Copyright    2022  Xiaomi Corp.        (authors: Wei Kang)
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

import math
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import dataclass, field
from zipformer.decode.context_graph import ContextState
from zipformer.decode.ngram_lm import NgramLmStateCost

from zipformer.utils.utils import AttributeDict


@dataclass
class KeywordResult:
    timestamps: List[int]
    hyps: List[int]
    phrase: str


@dataclass
class AsrResults:
    timestamps: List[List[int]]
    hyps: List[List[int]]
    scores: Optional[List[List[float]]] = None


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int] = field(default_factory=list)

    # The log prob of ys (used by transducer beam search).
    log_prob: Optional[torch.Tensor] = None

    # The lm score of ys
    # May contain external LM score (including LODR score) and contextual biasing score
    # It contains only one entry
    lm_score: torch.Tensor = torch.zeros(1, dtype=torch.float32)

    # used by keywords search. It stores the acoustic score for each token in ys.
    ac_probs: Optional[List[float]] = None

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamps: List[int] = field(default_factory=list)

    # the nnlm score for next token given the current ys
    # shape is (vocab_size,)
    nnlm_scores: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    nnlm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # LODR N-gram LM state
    lodr_state: Optional[NgramLmStateCost] = None

    # N-gram LM state (for shallow fusion)
    ngram_state: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    num_trailing_blanks: int = 0

    # CTC prefix beam search fields
    log_prob_blank: Optional[torch.Tensor] = None
    log_prob_non_blank: Optional[torch.Tensor] = None

    @property
    def tot_score(self) -> torch.Tensor:
        """Return the total score of this hypothesis.

        For CTC: logaddexp(log_prob_blank, log_prob_non_blank).
        For transducer: log_prob.
        """
        if self.log_prob_blank is not None:
            return torch.logaddexp(self.log_prob_non_blank, self.log_prob_blank)
        return self.log_prob

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))

    def clone(self) -> "Hypothesis":
        """Return a shallow clone (used by CTC prefix beam search)."""
        return Hypothesis(
            ys=self.ys,
            log_prob=self.log_prob,
            lm_score=self.lm_score,
            ac_probs=self.ac_probs,
            timestamps=self.timestamps,
            nnlm_scores=self.nnlm_scores,
            nnlm_states=self.nnlm_states,
            lodr_state=self.lodr_state,
            ngram_state=self.ngram_state,
            context_state=self.context_state,
            num_trailing_blanks=self.num_trailing_blanks,
            log_prob_blank=self.log_prob_blank,
            log_prob_non_blank=self.log_prob_non_blank,
        )


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
            if hyp.log_prob_blank is not None:
                # CTC prefix beam search: merge blank and non-blank separately
                torch.logaddexp(
                    old_hyp.log_prob_blank,
                    hyp.log_prob_blank,
                    out=old_hyp.log_prob_blank,
                )
                torch.logaddexp(
                    old_hyp.log_prob_non_blank,
                    hyp.log_prob_non_blank,
                    out=old_hyp.log_prob_non_blank,
                )
            else:
                torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest score.

        Args:
          length_norm:
            If True, the score of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest score.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score)

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
        """Remove all Hypotheses whose score is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with score being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.tot_score > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the score of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].tot_score / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].tot_score, reverse=True)[:k]

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


class DecodeStream(object):
    def __init__(
        self,
        params: AttributeDict,
        utt_id: str,
        initial_states: List[torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
          params:
            The decoding parameters.
          utt_id:
            The utterance id of this stream.
          initial_states:
            Initial decode states of the model, e.g. the return value of
            `get_init_state` in conformer.py
          device:
            The device to run this stream.
        """
        self.params = params
        self.utt_id = utt_id
        self.LOG_EPS = math.log(1e-10)

        self.states = initial_states

        # It contains a 2-D tensors representing the feature frames.
        self.features: torch.Tensor = None

        self.num_frames: int = 0
        # how many frames have been processed. (before subsampling).
        # we only modify this value in `func:get_feature_frames`.
        self.num_processed_frames: int = 0

        self._done: bool = False

        # The transcript of current utterance.
        self.ground_truth: str = ""

        # The decoding result (partial or final) of current utterance.
        self.hyp: List = []

        # how many frames have been processed, after subsampling (i.e. a
        # cumulative sum of the second return value of
        # encoder.streaming_forward
        self.done_frames: int = 0

        self.pad_length = (params.right_context + 2) * params.subsampling_factor + 3

        if params.decoding_method == "greedy_search":
            self.hyp = [params.blank_id] * params.context_size
        elif params.decoding_method == "modified_beam_search":
            self.hyps = HypothesisList()
            self.hyps.add(
                Hypothesis(
                    ys=[params.blank_id] * params.context_size,
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                )
            )
        else:
            raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    @property
    def done(self) -> bool:
        """Return True if all the features are processed."""
        return self._done

    @property
    def id(self) -> str:
        return self.utt_id

    def set_features(
        self,
        features: torch.Tensor,
    ) -> None:
        """Set features tensor of current utterance."""
        assert features.dim() == 2, features.dim()
        self.features = torch.nn.functional.pad(
            features,
            (0, 0, 0, self.pad_length),
            mode="constant",
            value=self.LOG_EPS,
        )
        self.num_frames = self.features.size(0)

    def get_feature_frames(self, chunk_size: int) -> Tuple[torch.Tensor, int]:
        """Consume chunk_size frames of features"""
        chunk_length = chunk_size + self.pad_length

        ret_length = min(self.num_frames - self.num_processed_frames, chunk_length)

        ret_features = self.features[
            self.num_processed_frames : self.num_processed_frames
            + ret_length  # noqa
        ]

        self.num_processed_frames += chunk_size
        if self.num_processed_frames >= self.num_frames:
            self._done = True

        return ret_features, ret_length

    def decoding_result(self) -> List[int]:
        """Obtain current decoding result."""
        if self.params.decoding_method == "greedy_search":
            return self.hyp[self.params.context_size :]  # noqa
        elif self.params.decoding_method == "modified_beam_search":
            best_hyp = self.hyps.get_most_probable(length_norm=True)
            return best_hyp.ys[self.params.context_size :]  # noqa
        else:
            raise ValueError(
                f"Unsupported decoding method: {self.params.decoding_method}"
            )
