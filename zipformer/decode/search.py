# Copyright    2021-2026  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Fangjun Kuang
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

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from zipformer.decode.context_graph import ContextGraph
from zipformer.decode.ngram_lm import NgramLm, NgramLmStateCost

from multiprocessing.pool import Pool
from zipformer.decode.stream import (
    DecodeStream,
    Hypothesis,
    HypothesisList,
    AsrResults,
    KeywordResult,
)


def _greedy_search_batch(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], AsrResults]:
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
      Else, return a AsrResults object containing
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
        return AsrResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )


def greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    max_sym_per_frame: int = 1,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], AsrResults]:
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
      Else, return a AsrResults object containing
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
    timestamps = []

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
            timestamps.append(t)
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
        return AsrResults(
            hyps=[hyp],
            timestamps=[timestamps],
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


def _score_nnlm(
    A: List[List[Hypothesis]],
    per_utt_topk: List[Tuple[torch.Tensor, torch.Tensor]],
    nnlm: torch.nn.Module,
    vocab_size: int,
    context_size: int,
    blank_id: int,
    unk_id: int,
    sos_id: int,
    device: torch.device,
) -> Tuple[List, List, List, int]:
    """First pass of two-pass LM scoring: collect non-blank tokens and score them.

    Returns (scores, lm_states, token_list, count_offset).
    Pass 1: iterate topk results, collect non-blank tokens for LM scoring.
    Pass 2 is done by the caller using the returned scores/lm_states.
    """
    if nnlm is None:
        return None, None
    token_list = []
    hs = []
    cs = []
    batch_size = len(per_utt_topk)
    for i in range(batch_size):
        _, topk_indexes = per_utt_topk[i]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
            topk_token_indexes = (topk_indexes % vocab_size).tolist()
        for k in range(len(topk_hyp_indexes)):
            hyp_idx = topk_hyp_indexes[k]
            hyp = A[i][hyp_idx]

            new_token = topk_token_indexes[k]
            if new_token not in (blank_id, unk_id):
                if nnlm.lm_type == "rnn":
                    token_list.append([new_token])
                    hs.append(hyp.state[0])
                    cs.append(hyp.state[1])
                else:
                    token_list.append([sos_id] + hyp.ys[context_size:] + [new_token])
    scores = None
    lm_states = None
    if len(token_list) != 0:
        x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
        if nnlm.lm_type == "rnn":
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
        scores, lm_states = nnlm.score_token(tokens_to_score, x_lens, state)
    return scores, lm_states


def modified_beam_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    context_graph: Optional[ContextGraph] = None,
    nnlm: Optional[torch.nn.Module] = None,
    nnlm_scale: float = 0.0,
    lodr: Optional[NgramLm] = None,
    lodr_scale: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], AsrResults, Dict]:
    """
    It implements a modified beam search with the following features:
      1) Support for contextual biasing using `context_graph`.
      2) Support for shallow fusion with an external NNLM `nnlm`.
      3) Support for LODR scoring with an N-gram LM `lodr`.

    Note: It assumes the --max-sym-per-frame to be set to 1.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the lengths of each sequence in the batch.
      beam:
        The beam size for beam search.
      temperature:
        The temperature for scaling the logits before softmax.
      blank_penalty:
        The score used to penalize blank probability.
      context_graph:
        An optional ContextGraph for contextual biasing.
      nnlm:
        An optional neural network language model for shallow fusion.
      nnlm_scale:
        The scale for shallow fusion with the NNLM.
      lodr:
        An optional N-gram LM for LODR scoring.
      lodr_scale:
        The scale for LODR scoring.
      return_timestamps:
        Whether to return timestamps.
        If True, the returned AsrResults will contain timestamps for each predicted token.

    Returns:
        If return_timestamps is False, return a list of decoded results (list of token ids)
        for each utterance in the batch. Else, return an AsrResults object containing decoded results
        and corresponding timestamps for each utterance in the batch.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
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
    sos_id = getattr(nnlm, "sos_id", 1) if nnlm is not None else 1
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    init_scores, init_states = None, None
    if nnlm is not None:
        # get initial nnlm score and states by scoring "sos" token
        sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
        lens = torch.tensor([1]).to(device)
        init_scores, init_states = nnlm.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        hyp_kwargs = dict(
            ys=[-1] * (context_size - 1) + [blank_id],
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            timestamps=[],
        )
        if context_graph is not None:
            hyp_kwargs["context_state"] = context_graph.root
        if lodr is not None:
            hyp_kwargs["lodr_state"] = NgramLmStateCost(lodr)
        if nnlm is not None:
            hyp_kwargs["nnlm_states"] = init_states
            hyp_kwargs["nnlm_scores"] = init_scores.reshape(-1)
        B[i].add(Hypothesis(**hyp_kwargs))

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

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

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (total_num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids().to(torch.int64).to(device),
        )  # (total_num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (total_num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (total_num_hyps, vocab_size)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        log_probs = (logits / temperature).log_softmax(dim=-1)
        log_probs.add_(ys_log_probs)
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits() * vocab_size
        per_utt_topk = _per_utterance_topk(log_probs, row_splits, beam)

        # Scoring nnlm for shallow fusion
        new_nnlm_scores, new_nnlm_states = _score_nnlm(
            hyps=A,
            per_utt_topk=per_utt_topk,
            nnlm=nnlm,
            vocab_size=vocab_size,
            context_size=context_size,
            blank_id=blank_id,
            unk_id=unk_id,
            sos_id=sos_id,
            device=device,
        )

        count = 0  # for nnlm scoring index
        for i in range(batch_size):
            topk_log_probs, topk_indexes = per_utt_topk[i]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]
                token = topk_token_indexes[k]
                timestamps = hyp.timestamps[:] if hasattr(hyp, "timestamps") else []
                nnlm_scores = hyp.nnlm_scores if hasattr(hyp, "nnlm_scores") else None
                nnlm_states = hyp.nnlm_states if hasattr(hyp, "nnlm_states") else None
                lodr_state = hyp.lodr_state if hasattr(hyp, "lodr_state") else None
                hyp_log_prob = topk_log_probs[k]
                context_score = 0
                context_state = None if context_graph is None else hyp.context_state

                if token not in (blank_id, unk_id):
                    ys.append(token)
                    timestamps.append(t)
                    # context graph scoring
                    if context_graph is not None:
                        (
                            context_score,
                            context_state,
                            _,
                        ) = context_graph.forward_one_step(
                            context_state, token, strict_mode=False
                        )
                    current_lodr_score = 0
                    if lodr is not None:
                        lodr_state = hyp.lodr_state.forward_one_step(token)
                        current_lodr_score = (
                            lodr_state.lm_score - hyp.lodr_state.lm_score
                        )
                        assert current_lodr_score <= 0.0, (
                            lodr_state.lm_score,
                            hyp.lodr_state.lm_score,
                        )

                    current_nnlm_score = (
                        0 if nnlm_scores is None else nnlm_scores[token]
                    )
                    # lodr_scale should be a negative number
                    hyp_log_prob += (
                        current_nnlm_score * nnlm_scale
                        + lodr_scale * current_lodr_score
                        + context_score
                    )

                    if new_nnlm_scores is not None:
                        nnlm_scores = new_nnlm_scores[count]
                        if nnlm.lm_type == "rnn":
                            nnlm_states = (
                                new_nnlm_states[0][:, count, :].unsqueeze(1),
                                new_nnlm_states[1][:, count, :].unsqueeze(1),
                            )
                        count += 1

                build_kwargs = dict(ys=ys, log_prob=hyp_log_prob, timestamps=timestamps)
                if lodr is not None:
                    build_kwargs["lodr_state"] = lodr_state
                elif nnlm is not None:
                    build_kwargs["nnlm_states"] = nnlm_states
                    build_kwargs["nnlm_scores"] = nnlm_scores
                elif context_graph is not None:
                    build_kwargs["context_state"] = context_state

                new_hyp = Hypothesis(**build_kwargs)
                B[i].add(new_hyp)

    B = B + finalized_B

    if context_graph is not None:
        for hyps in B:
            for hyp in hyps:
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                hyp.lm_score += context_score
                hyp.context_state = new_context_state

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [getattr(h, "timestamps", []) for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])
    if not return_timestamps:
        return ans
    else:
        return AsrResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )


def beam_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], AsrResults]:
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
            ys=[-1] * (context_size - 1) + [blank_id], log_prob=0.0, timestamps=[]
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
                    timestamps=y_star.timestamps[:],
                )
            )

            # Second, process other non-blank labels
            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i in (blank_id, unk_id):
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                new_timestamps = y_star.timestamps + [t]
                A.add(
                    Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        timestamps=new_timestamps,
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
        return AsrResults(hyps=[ys], timestamps=[best_hyp.timestamps])


def streaming_greedy_search(
    model: torch.nn.Module,
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
    model: torch.nn.Module,
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
      blank_penalty:
        The score used to penalize blank probability.
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
            index=hyps_shape.row_ids().to(torch.int64),
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

        row_splits = hyps_shape.row_splits() * vocab_size
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


def _step_worker(
    log_probs: torch.Tensor,
    indexes: torch.Tensor,
    B: HypothesisList,
    beam: int = 4,
    blank_id: int = 0,
    nnlm_scale: float = 0,
    lodr_scale: float = 0,
    context_graph: Optional[ContextGraph] = None,
) -> HypothesisList:
    """The worker to decode one step.
    Args:
      log_probs:
        topk log_probs of current step (i.e. the kept tokens of first pass pruning),
        the shape is (beam,)
      topk_indexes:
        The indexes of the topk_values above, the shape is (beam,)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      nnlm_scale:
        The scale of nn lm.
      lodr_scale:
        The scale of the LODR_lm
      context_graph:
        A ContextGraph instance containing contextual phrases.
    Return:
      Returns the updated HypothesisList.
    """
    A = list(B)
    B = HypothesisList()
    for h in range(len(A)):
        hyp = A[h]
        for k in range(log_probs.size(0)):
            log_prob, index = log_probs[k], indexes[k]
            new_token = index.item()
            update_prefix = False
            new_hyp = hyp.clone()
            if new_token == blank_id:
                # Case 0: *a + ε => *a
                #         *aε + ε => *a
                # Prefix does not change, update log_prob of blank
                new_hyp.log_prob_non_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                new_hyp.log_prob_blank = hyp.log_prob + log_prob
                B.add(new_hyp)
            elif len(hyp.ys) > 0 and hyp.ys[-1] == new_token:
                # Case 1: *a + a => *a
                # Prefix does not change, update log_prob of non_blank
                new_hyp.log_prob_non_blank = hyp.log_prob_non_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                B.add(new_hyp)

                # Case 2: *aε + a => *aa
                # Prefix changes, update log_prob of blank
                new_hyp = hyp.clone()
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True
            else:
                # Case 3: *a + b => *ab, *aε + b => *ab
                # Prefix changes, update log_prob of non_blank
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True

            if update_prefix:
                lm_score = hyp.lm_score
                if hyp.nnlm_scores is not None:
                    lm_score = lm_score + hyp.nnlm_scores[new_token] * nnlm_scale
                    new_hyp.nnlm_scores = None

                if context_graph is not None and hyp.context_state is not None:
                    (
                        context_score,
                        new_context_state,
                        matched_state,
                    ) = context_graph.forward_one_step(hyp.context_state, new_token)
                    lm_score = lm_score + context_score
                    new_hyp.context_state = new_context_state

                if hyp.lodr_state is not None:
                    lodr_state = hyp.lodr_state.forward_one_step(new_token)
                    # calculate the score of the latest token
                    current_lodr_score = lodr_state.lm_score - hyp.lodr_state.lm_score
                    assert current_lodr_score <= 0.0, (
                        lodr_state.lm_score,
                        hyp.lodr_state.lm_score,
                    )
                    lm_score = lm_score + lodr_scale * current_lodr_score
                    new_hyp.lodr_state = lodr_state

                new_hyp.lm_score = lm_score
                B.add(new_hyp)
    B = B.topk(beam)
    return B


def _sequence_worker(
    topk_values: torch.Tensor,
    topk_indexes: torch.Tensor,
    B: HypothesisList,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
) -> HypothesisList:
    """The worker to decode one sequence.
    Args:
      topk_values:
        topk log_probs of model output (i.e. the kept tokens of first pass pruning),
        the shape is (T, beam)
      topk_indexes:
        The indexes of the topk_values above, the shape is (T, beam)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
    Return:
      Returns the updated HypothesisList.
    """
    B.add(Hypothesis())
    for j in range(encoder_out_lens):
        log_probs, indexes = topk_values[j], topk_indexes[j]
        B = _step_worker(log_probs, indexes, B, beam, blank_id)
    return B


def _ctc_prefix_beam_search_parallel(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    process_pool: Optional[Pool] = None,
    return_nbest: Optional[bool] = False,
) -> Union[List[List[int]], List[HypothesisList]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks".
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
      return_nbest:
        If true, return a list of HypothesisList, return a list of list of decoded token ids otherwise.
    """
    batch_size, num_frames, vocab_size = ctc_output.shape

    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()

    B = [HypothesisList() for _ in range(batch_size)]

    pool = Pool() if process_pool is None else process_pool
    arguments = []
    for i in range(batch_size):
        arguments.append(
            (
                topk_values[i],
                topk_indexes[i],
                B[i],
                encoder_out_lens[i].item(),
                beam,
                blank_id,
            )
        )
    async_results = pool.starmap_async(_sequence_worker, arguments)
    B = list(async_results.get())
    if process_pool is None:
        pool.close()
        pool.join()
    if return_nbest:
        return B
    else:
        best_hyps = [b.get_most_probable() for b in B]
        return [hyp.ys for hyp in best_hyps]


def ctc_prefix_beam_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    lodr: Optional[NgramLm] = None,
    lodr_scale: Optional[float] = 0,
    nnlm: Optional[torch.nn.Module] = None,
    nnlm_scale: Optional[float] = 0,
    context_graph: Optional[ContextGraph] = None,
    process_pool: Optional[Pool] = None,
    return_nbest: Optional[bool] = False,
) -> List[List[int]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks" and add
    nervous language model shallow fussion, it also supports contextual
    biasing with a given grammar.
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      lodr:
        A low order n-gram LM, whose score will be subtracted during shallow fusion
      lodr_scale:
        The scale of the lodr
      nnlm:
        A neural net LM, e.g an RNNLM or transformer LM
      nnlm_scale:
        The scale of the nnlm
      context_graph:
        A ContextGraph instance containing contextual phrases.
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
      return_nbest:
        If true, return a list of HypothesisList, return a list of list of decoded token ids otherwise.

    Return:
      Returns a list of list of decoded token ids.
    """
    if lodr is None and nnlm is None and context_graph is None:
        return _ctc_prefix_beam_search_parallel(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            beam=beam,
            blank_id=blank_id,
            process_pool=process_pool,
            return_nbest=return_nbest,
        )
    batch_size, num_frames, vocab_size = ctc_output.shape
    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()
    encoder_out_lens = encoder_out_lens.tolist()
    device = ctc_output.device

    nnlm_scale = 0
    init_scores = None
    init_states = None
    if nnlm is not None:
        sos_id = getattr(nnlm, "sos_id", 1)
        # get initial lm score and lm state by scoring the "sos" token
        sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
        lens = torch.tensor([1]).to(device)
        init_scores, init_states = nnlm.score_token(sos_token, lens)
        init_scores, init_states = (
            init_scores.cpu(),
            (
                init_states[0].cpu(),
                init_states[1].cpu(),
            ),
        )

    B = [HypothesisList() for _ in range(batch_size)]
    for i in range(batch_size):
        B[i].add(
            Hypothesis(
                ys=[],
                log_prob_non_blank=torch.tensor([float("-inf")], dtype=torch.float32),
                log_prob_blank=torch.zeros(1, dtype=torch.float32),
                lm_score=torch.zeros(1, dtype=torch.float32),
                nnlm_states=init_states,
                nnlm_scores=None if init_scores is None else init_scores.reshape(-1),
                lodr_state=None if lodr is None else NgramLmStateCost(lodr),
                context_state=None if context_graph is None else context_graph.root,
            )
        )
    for j in range(num_frames):
        for i in range(batch_size):
            if j < encoder_out_lens[i]:
                log_probs, indexes = topk_values[i][j], topk_indexes[i][j]
                B[i] = _step_worker(
                    log_probs=log_probs,
                    indexes=indexes,
                    B=B[i],
                    beam=beam,
                    blank_id=blank_id,
                    nnlm_scale=nnlm_scale,
                    lodr_scale=lodr_scale,
                    context_graph=context_graph,
                )
        if nnlm is None:
            continue
        # update lm_log_probs
        token_list = []  # a list of list
        hs = []
        cs = []
        indexes = []  # (batch_idx, key)
        for batch_idx, hyps in enumerate(B):
            for hyp in hyps:
                if hyp.nnlm_scores is None:  # those hyps that prefix changes
                    if nnlm.lm_type == "rnn":
                        token_list.append([hyp.ys[-1]])
                        # store the LSTM states
                        hs.append(hyp.nnlm_states[0])
                        cs.append(hyp.nnlm_states[1])
                    else:
                        # for transformer LM
                        token_list.append([sos_id] + hyp.ys[:])
                    indexes.append((batch_idx, hyp.key))
        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if nnlm.lm_type == "rnn":
                tokens_to_score = (
                    torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
                )
                hs = torch.cat(hs, dim=1).to(device)
                cs = torch.cat(cs, dim=1).to(device)
                state = (hs, cs)
            else:
                # for transformer LM
                tokens_list = [torch.tensor(tokens) for tokens in token_list]
                tokens_to_score = (
                    torch.nn.utils.rnn.pad_sequence(
                        tokens_list, batch_first=True, padding_value=0.0
                    )
                    .to(device)
                    .to(torch.int64)
                )
                state = None

            scores, lm_states = nnlm.score_token(tokens_to_score, x_lens, state)
            scores, lm_states = scores.cpu(), (lm_states[0].cpu(), lm_states[1].cpu())
            assert scores.size(0) == len(indexes), (scores.size(0), len(indexes))
            for i in range(scores.size(0)):
                batch_idx, key = indexes[i]
                B[batch_idx][key].nnlm_scores = scores[i]
                if nnlm.lm_type == "rnn":
                    state = (
                        lm_states[0][:, i, :].unsqueeze(1),
                        lm_states[1][:, i, :].unsqueeze(1),
                    )
                    B[batch_idx][key].nnlm_states = state

    # finalize context_state, if the matched contexts do not reach final state
    # we need to add the score on the corresponding backoff arc
    if context_graph is not None:
        for hyps in B:
            for hyp in hyps:
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                hyp.lm_score += context_score
                hyp.context_state = new_context_state

    best_hyps = [b.get_most_probable() for b in B]
    return [hyp.ys for hyp in best_hyps]


# transduer keywords decoding.
def keywords_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    keywords_graph: ContextGraph,
    beam: int = 4,
    num_trailing_blanks: int = 0,
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
      num_trailing_blanks:
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
                timestamps=[],
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
            index=hyps_shape.row_ids().to(torch.int64),
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

        row_splits = hyps_shape.row_splits() * vocab_size
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
                new_timestamps = hyp.timestamps[:]
                new_ac_probs = hyp.ac_probs[:]
                context_score = 0
                new_context_state = hyp.context_state
                new_num_trailing_blanks = hyp.num_trailing_blanks + 1
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamps.append(t)
                    new_ac_probs.append(hyp_probs[topk_indexes[k]])
                    (
                        context_score,
                        new_context_state,
                        _,
                    ) = keywords_graph.forward_one_step(hyp.context_state, new_token)
                    new_num_trailing_blanks = 0
                    if new_context_state.token == -1:  # root
                        new_ys[-context_size:] = [-1] * (context_size - 1) + [blank_id]

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamps=new_timestamps,
                    ac_probs=new_ac_probs,
                    context_state=new_context_state,
                    num_trailing_blanks=new_num_trailing_blanks,
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
                and top_hyp.num_trailing_blanks > num_trailing_blanks
                and ac_prob >= matched_state.ac_threshold
            ):
                keyword = KeywordResult(
                    hyps=top_hyp.ys[-matched_state.level :],
                    timestamps=top_hyp.timestamps[-matched_state.level :],
                    phrase=matched_state.phrase,
                )
                sorted_ans[i].append(keyword)
                B[i] = HypothesisList()
                B[i].add(
                    Hypothesis(
                        ys=[-1] * (context_size - 1) + [blank_id],
                        log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                        context_state=keywords_graph.root,
                        timestamps=[],
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
                timestamps=top_hyp.timestamps[-matched_state.level :],
                phrase=matched_state.phrase,
            )
            sorted_ans[i].append(keyword)

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans
