# Copyright    2022-2026  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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

conversational_filler = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
]
unk_tags = ["<UNK>", "<unk>"]
gigaspeech_punctuations = [
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
]
gigaspeech_garbage_utterance_tags = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
non_scoring_words = (
    conversational_filler
    + unk_tags
    + gigaspeech_punctuations
    + gigaspeech_garbage_utterance_tags
)


def gigaspeech_post_processing(text: str) -> str:
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace("-", " ")

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return " ".join(remaining_words).lower()
