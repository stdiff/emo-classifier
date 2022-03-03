from importlib import resources
from collections import OrderedDict
from abc import ABC, abstractmethod
import json

import pandas as pd
from en_core_web_md import load as spacy_load
from torchtext.vocab import Vocab, vocab

from emo_classifier import setup_logger
from emo_classifier import ARTIFACT_DIR

logger = setup_logger(__name__)
vocab_file_name: str = "vocab.json"


class Tokenizer(ABC):
    """
    logic converting a text into list of tokens
    """

    @abstractmethod
    def __call__(self, text: str) -> list[str]:
        raise NotImplementedError


class SpacyEnglishTokenizer(Tokenizer):
    def __init__(self, with_lemmatization: bool = False, remove_stopwords: bool = True):
        if with_lemmatization:
            exclude = ["tok2vec", "parser", "ner"]
        else:
            exclude = ["tok2vec", "tagger", "parser", "attribute_ruler", "ner", "lemmatizer"]

        self.remove_stopwords = remove_stopwords
        self.nlp = spacy_load(exclude=exclude)

    @staticmethod
    def doc2tokens(doc) -> list[str]:
        return [token.lower_ for token in doc if not token.is_stop and not token.is_punct]

    def __call__(self, text: str) -> list[str]:
        if self.remove_stopwords:
            return self.doc2tokens(self.nlp(text))
        else:
            return [token.lower_ for token in self.nlp(text)]

    def tokenize_in_batch(self, s_text: pd.Series) -> pd.Series:
        """No significant improvement if with_lemmtatization is False"""
        return pd.Series(self.nlp.pipe(s_text), index=s_text.index, name="tokens").apply(self)


def save_vocab(vocab: Vocab):
    tokens = vocab.get_itos()
    file_path = ARTIFACT_DIR / vocab_file_name
    with file_path.open("w") as fp:
        json.dump(tokens, fp)
    logger.info(f"SAVED: {file_path}")


def load_vocab() -> Vocab:
    with resources.open_text("emo_classifier.artifact", vocab_file_name) as fp:
        tokens = json.load(fp)

    logger.info(f"Vocabulary vocab size = {len(tokens)}")
    return vocab(OrderedDict([(token, 1) for token in tokens]))
