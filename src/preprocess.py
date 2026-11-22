"""NLP preprocessing utilities built on NLTK.

Pipeline: expand contractions -> normalize -> tokenize -> remove
stopwords/punctuation -> POS-aware lemmatization. The same function is used
for training and inference so the model always sees text in the same form.
"""

import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

_CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _CONTRACTIONS)) + r")\b", re.IGNORECASE
)

# Question words and negations carry intent, so they are kept as features.
_INTENT_STOPWORDS = {
    "no",
    "not",
    "nor",
    "never",
    "cannot",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "my",
    "your",
    "our",
}

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english")) - _INTENT_STOPWORDS


def ensure_nltk_data() -> None:
    """Download required NLTK corpora only when they are missing."""
    for name, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()


def _expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(
        lambda match: _CONTRACTIONS[match.group(0).lower()], text
    )


def normalize_text(text: str) -> str:
    """Lowercase and strip URLs, dataset placeholders and punctuation."""
    text = str(text)
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _expand_contractions(text).lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text with NLTK's word tokenizer."""
    return word_tokenize(normalize_text(text))


def _wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using their part-of-speech tags."""
    if not tokens:
        return []
    tagged = nltk.pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged
    ]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline returning a cleaned string."""
    tokens = [
        token
        for token in tokenize(text)
        if token.isalpha() and len(token) > 1 and token not in _STOPWORDS
    ]
    return " ".join(lemmatize(tokens))
