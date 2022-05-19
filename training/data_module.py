from typing import Optional, Iterator, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab
import pytorch_lightning as pl

from emo_classifier.classifiers.text import SpacyEnglishTokenizer, load_vocab
from emo_classifier.classifiers.embedding_bag import padding_token, unknown_token
from emo_classifier.emotion import load_emotions
from training.preprocessing import Preprocessor


class GoEmotionsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        with_lemmatization: bool = False,
        remove_stopwords: bool = False,
        min_df: int = 10,
        input_length: int = 10,
        batch_size: int = 32,
        batch_size_eval: int = 128,
        load_vocab: bool = False,
    ):
        """

        :param with_lemmatization: whether a tokenizer does lemmatization
        :param remove_stopwords: whether a tokenizer removes stopwords
        :param min_df: minimum document frequency (DF). A term with lower DF will be ignored.
        :param input_length: maximum number of tokens for the input of the model
        :param batch_size: batch size for training
        :param batch_size_eval: batch size for evaluation (validation, test)
        :param load_vocab: if a Vocab instance will be loaded.
        """
        super().__init__()
        self.with_lemmatization = with_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_df = min_df

        self.tokenizer = SpacyEnglishTokenizer(self.with_lemmatization, self.remove_stopwords)
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.load_vocab = load_vocab
        self.vocab: Optional[Vocab] = None
        self.padding_index = 0
        self.input_length = input_length

        self.preprocessor: Optional[Preprocessor] = None
        self.ds_train: Optional[TensorDataset] = None
        self.ds_val: Optional[TensorDataset] = None
        self.ds_dev: Optional[TensorDataset] = None
        self.ds_test: Optional[TensorDataset] = None
        self.emotions: list[str] = load_emotions()

        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

    def build_vocab(self, texts: Iterator[str], min_freq: int = 20):
        if self.load_vocab:
            self.vocab = load_vocab()
        else:
            self.vocab = build_vocab_from_iterator(
                (self.tokenizer(text) for text in texts),
                specials=[self.padding_token, self.unknown_token],
                min_freq=min_freq,
            )

        self.vocab.set_default_index(self.vocab[self.unknown_token])
        self.padding_index = self.vocab[self.padding_token]

    def texts2tensor(self, texts: Iterable[str]) -> torch.Tensor:
        sequences_of_indices = [torch.tensor(self.vocab(self.tokenizer(text))) for text in texts]
        X = pad_sequence(sequences_of_indices, batch_first=True, padding_value=self.padding_index)

        if X.shape[1] >= self.input_length:
            return X[:, : self.input_length]
        else:
            Z = torch.zeros((X.shape[0], self.input_length - X.shape[1]), dtype=torch.int64)
            return torch.hstack((X, Z))

    def XY2TensorDataset(self, X: pd.Series, Y: pd.DataFrame) -> TensorDataset:
        X_tensor = self.texts2tensor(X)
        Y_tensor = torch.from_numpy(Y.values).float()
        return TensorDataset(X_tensor, Y_tensor)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.preprocessor = Preprocessor()
            X_train, Y_train = self.preprocessor.get_train_X_and_Y()
            self.build_vocab(X_train)

            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=4000, random_state=9)
            self.ds_train = self.XY2TensorDataset(X_train, Y_train)
            self.ds_val = self.XY2TensorDataset(X_val, Y_val)

        elif stage == "test":
            X_dev, Y_dev = self.preprocessor.get_dev_X_and_Y()
            self.ds_dev = self.XY2TensorDataset(X_dev, Y_dev)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_val, batch_size=self.batch_size_eval, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_dev, batch_size=self.batch_size_eval, shuffle=False)
