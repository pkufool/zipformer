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
