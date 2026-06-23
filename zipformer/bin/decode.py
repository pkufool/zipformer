#!/usr/bin/env python3
# Copyright 2021-2026 Xiaomi Corporation (Author: Wei Kang,
#                                                 Zengwei Yao,
#                                                 Fangjun Kuang,
#                                                 Zengrui Jin)
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

"""
Combined CTC and RNN-T decoding script
Supports a minimal set of decoding methods:
    - rnnt-greedy-search
    - rnnt-modified-beam-search
    - ctc-greedy-search
    - ctc-prefix-beam-search
"""

import argparse
import logging

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ssentencepiece import Ssentencepiece
from tqdm import tqdm
from atdataset import ATDataloader, Fbank

from zipformer.bin.train import add_model_arguments, get_model, get_params
from zipformer.decode.search import (
    greedy_search,
    modified_beam_search,
    ctc_greedy_search,
    ctc_prefix_beam_search,
)
from zipformer.decode.post_processing import gigaspeech_post_processing
from zipformer.utils import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
    AttributeDict,
    LOG_EPS,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
    replace_punctuation_with_space,
    tokenize_by_cjk_char,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="rnnt-greedy-search",
        choices=sorted(
            [
                "rnnt-greedy-search",
                "rnnt-modified-beam-search",
                "ctc-greedy-search",
                "ctc-prefix-beam-search",
            ]
        ),
        help="Choose between rnnt or ctc decoding variants",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size used by rnnt modified beam search and ctc prefix beam search",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="Maximum symbols per frame for rnnt greedy search",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=600.0,
        help="Maximum duration (s) for each cut",
    )

    parser.add_argument(
        "--test-sets",
        nargs="+",
        help="""
        A list of test sets, each item contains the test set name and manifest path,e.g., 
        dev,data/tars/librispeech_dev.lst, test_other,data/tars/librispeech_test-other.lst
        """,
    )

    parser.add_argument(
        "--ignore-punctuation",
        type=str2bool,
        default=True,
        help="Whether to remove punctuation when calculating WER",
    )

    parser.add_argument(
        "--gigaspeech-post-processing",
        type=str2bool,
        default=False,
        help="Whether to apply GigaSpeech-specific post-processing to the reference and hypothesis texts before WER calculation.",
    )

    add_model_arguments(parser)
    return parser


def _load_checkpoint(
    params: AttributeDict, model: torch.nn.Module, device: torch.device
) -> None:
    """Load checkpoints with or without averaging."""
    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            if len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = [
                f"{params.exp_dir}/epoch-{i}.pt"
                for i in range(start, params.epoch + 1)
                if i >= 1
            ]
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            if len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints "
                f"from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0
            start = params.epoch - params.avg
            assert start >= 1
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from {start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )


def decode_batch_rnnt(
    params: AttributeDict,
    model: torch.nn.Module,
    sp: Ssentencepiece,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """
    Perform decoding for a batch of data using RNN-T decoding methods.
    Depending on the specified decoding method in params, it can perform either
    greedy search or modified beam search. The function returns a dictionary where
    the keys are the decoding method names and the values are lists of hypotheses,
    with each hypothesis being a list of decoded words.

    Args:
        params: An AttributeDict containing decoding parameters, including the decoding method and beam size.
        model: The RNN-T model to be used for decoding.
        sp: The sentencepiece model for decoding token IDs to words.
        batch: A dictionary containing the input features and their lengths for the batch.

    Returns:
        A dictionary mapping decoding method names to lists of decoded hypotheses. Each hypothesis is a list of decoded words.
    """
    device = next(model.parameters()).device
    feature = batch["feature"].to(device)
    feature_lens = batch["feature_lens"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        feature = torch.nn.functional.pad(
            feature, pad=(0, 0, 0, pad_len), value=LOG_EPS
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    if params.decoding_method == "rnnt-greedy-search":
        hyp_tokens = greedy_search(
            model=model, encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
        )
        hyps = [sp.decode(hyp).split() for hyp in hyp_tokens]
        return {"rnnt-greedy-search": hyps}

    assert params.decoding_method == "rnnt-modified-beam-search"
    hyp_tokens = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
    )
    hyps = [sp.decode(hyp).split() for hyp in hyp_tokens]
    return {f"rnnt-modified-beam-search_beam-{params.beam_size}": hyps}


def decode_batch_ctc(
    params: AttributeDict,
    model: torch.nn.Module,
    sp: Ssentencepiece,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """
    Perform decoding for a batch of data using CTC decoding methods.
    Depending on the specified decoding method in params, it can perform either
    greedy search or prefix beam search. The function returns a dictionary where
    the keys are the decoding method names and the values are lists of hypotheses,
    with each hypothesis being a list of decoded words.

    Args:
        params: An AttributeDict containing decoding parameters, including the decoding method and beam size.
        model: The CTC model to be used for decoding.
        sp: The sentencepiece model for decoding token IDs to words.
        batch: A dictionary containing the input features and their lengths for the batch.

    Returns:
        A dictionary mapping decoding method names to lists of decoded hypotheses. Each hypothesis is a list of decoded words.
    """
    device = params.device
    feature = batch["feature"].to(device)
    feature_lens = batch["feature_lens"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        feature = torch.nn.functional.pad(
            feature, pad=(0, 0, 0, pad_len), value=LOG_EPS
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
    ctc_output = model.ctc_output(encoder_out)

    if params.decoding_method == "ctc-greedy-search":
        token_ids = ctc_greedy_search(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            blank_id=params.blank_id,
        )
        hyps = [sp.decode(t).split() for t in token_ids]
        return {"ctc-greedy-search": hyps}

    assert params.decoding_method == "ctc-prefix-beam-search"
    token_ids = ctc_prefix_beam_search(
        ctc_output=ctc_output,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
        blank_id=params.blank_id,
    )
    hyps = [sp.decode(t).split() for t in token_ids]
    return {f"ctc-prefix-beam-search_beam-{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: torch.nn.Module,
    sp: Ssentencepiece,
    decode_fn,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """
    Decode an entire dataset using the provided dataloader and decoding function.
    This function iterates through the batches of data provided by the dataloader,
    applies the decoding function to each batch, and collects the results in a dictionary.
    The results dictionary maps decoding method names to lists of tuples, where each tuple contains the cut ID, reference words, and hypothesis words.

    Args:
        dl: A DataLoader providing batches of data to decode.
        params: An AttributeDict containing decoding parameters.
        model: The model to be used for decoding.
        sp: The sentencepiece model for decoding token IDs to words.
        decode_fn: The decoding function to apply to each batch.

    Returns:
        A dictionary mapping decoding method names to lists of tuples, where each tuple contains the cut ID, reference words, and hypothesis words.
    """
    results = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(dl, total=len(dl))):
        texts = batch["text"]
        utt_ids = batch["ids"]
        if not utt_ids:
            utt_ids = list(range(len(texts)))

        hyps_dict = decode_fn(params=params, model=model, sp=sp, batch=batch)

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for utt_id, hyp_words, ref_text in zip(utt_ids, hyps, texts):
                ref_words = ref_text.strip()
                hyp_words = " ".join(hyp_words).strip()

                if params.gigaspeech_post_processing:
                    ref_words = gigaspeech_post_processing(ref_words)
                    hyp_words = gigaspeech_post_processing(hyp_words)

                if params.ignore_punctuation:
                    ref_words = replace_punctuation_with_space(ref_words)
                    hyp_words = replace_punctuation_with_space(hyp_words)

                ref_words = tokenize_by_cjk_char(ref_words.lower())
                hyp_words = tokenize_by_cjk_char(hyp_words.lower())

                this_batch.append((utt_id, ref_words, hyp_words))
            results[name].extend(this_batch)

    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save the ASR output for each decoding method to files.
    For each decoding method, it writes the recognized transcripts to a file named `recogs-{test_set_name}-{params.suffix}.txt`
    and logs the location of the saved transcripts.
    """
    for key, results in results_dict.items():
        recogs_filename = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recogs_filename, texts=results)
        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Calculate WER for each decoding method and save the results to files.
    For each decoding method, it writes detailed error statistics to a file named `errs-{test_set_name}-{params.suffix}.txt`
    and logs the WER. It also creates a summary file named `wer-summary-{test_set_name}-{params.suffix}.txt` that lists
    the WER for each decoding method in a tab-separated format. Finally, it logs a summary of the WER results for the test set.
    """
    test_set_wers = dict()
    for key, results in results_dict.items():
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer
        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    wer_filename = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(wer_filename, "w", encoding="utf8") as fd:
        print("settings\tWER", file=fd)
        for key, val in test_set_wers:
            print(f"{key}\t{val}", file=fd)

    summary = f"\nFor {test_set_name}, WER of different settings are:\n"
    note = f"\tbest for {test_set_name}"
    for key, val in test_set_wers:
        summary += f"{key}\t{val}{note}\n"
        note = ""
    logging.info(summary)


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / params.decoding_method
    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    if params.causal:
        assert "," not in params.chunk_size, (
            "chunk_size should be one value in decoding."
        )
        assert "," not in params.left_context_frames, (
            "left_context_frames should be one value in decoding."
        )
        params.suffix += (
            f"_chunk-{params.chunk_size}_left-context-{params.left_context_frames}"
        )

    if "beam" in params.decoding_method:
        params.suffix += f"_beam-size-{params.beam_size}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    params.device = device

    sp = Ssentencepiece(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos>")
    params.vocab_size = sp.vocab_size()

    logging.info(f"Decoding started: {params}")

    logging.info("About to create model")
    model = get_model(params)
    _load_checkpoint(params, model, device)
    model.to(device)
    model.eval()

    assert params.test_sets is not None and len(params.test_sets) > 0, (
        "Please specify test sets for decoding."
    )

    decode_fn = (
        decode_batch_ctc
        if params.decoding_method.startswith("ctc-")
        else decode_batch_rnnt
    )

    test_sets = dict()
    for item in params.test_sets:
        key, path = item.split(",")
        test_sets[key] = path

    feature_extractor = Fbank(sample_rate=params.sample_rate, n_mels=params.feature_dim)

    for test_set, manifest in test_sets.items():
        test_dl = ATDataloader(
            datasets=manifest,
            max_duration=params.max_duration,
            feature_extractor=feature_extractor,
            sample_rate=params.sample_rate,
            is_test=True,
            num_workers=0,
        )

        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            decode_fn=decode_fn,
        )

        save_asr_output(
            params=params, test_set_name=test_set, results_dict=results_dict
        )
        save_wer_results(
            params=params, test_set_name=test_set, results_dict=results_dict
        )
    logging.info("Done!")


if __name__ == "__main__":
    main()
