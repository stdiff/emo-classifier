from typing import Optional, Iterable, Union
from importlib import resources
from abc import ABC, abstractmethod

import joblib
import pandas as pd
from en_core_web_md import load as spacy_load
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator

from emo_classifier import ARTIFACT_DIR


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


# class TextNumerator:
#     """
#     manages
#     - tokenizer
#     - vocabulary
#     - numericalizing of tokens (list of tokens -> list of indexes)
#     """
#
#     def __init__(self, with_lemmatization: bool = False, min_df: int = 10, remove_stopwords: bool = False):
#         self.with_lemmatization = with_lemmatization
#         self.min_df = min_df
#         self.remove_stopwords = remove_stopwords
#
#         self.tokenizer = Tokenizer(with_lemmatization=self.with_lemmatization, remove_stopwords=self.remove_stopwords)
#         self.empty_token = ""
#         self.unknown_token = "<unk>"
#
#         self.vocab: Optional[Vocab] = None
#
#     def save(self):
#         data = {"params": {"with_lemmatization": self.with_lemmatization, "min_df": self.min_df}, "vocab": self.vocab}
#         file_path = ARTIFACT_DIR / f"{type(self).__name__}.joblib"
#         joblib.dump(data, file_path)
#
#     @classmethod
#     def load(cls) -> "TextNumerator":
#         with resources.open_binary("emo_classifier.artifact", f"{cls.__name__}.joblib") as fo:
#             data = joblib.load(fo)
#
#         text_numerator = cls(**data["params"])
#         text_numerator.vocab = data["vocab"]
#         return text_numerator
#
#     def build_vocab(self, s_test: pd.Series):
#         self.vocab = build_vocab_from_iterator(
#             self.tokenizer.tokenize_in_batch(s_test),
#             specials=[self.empty_token, self.unknown_token],
#             min_freq=self.min_df,
#         )
#         self.vocab.set_default_index(self.vocab[self.unknown_token])
#
#     def __call__(self, text: Union[str, Iterable[str]]):
#         tokens = self.tokenizer(text)
#         return numericalize_tokens_from_iterator(self.vocab, [tokens])
