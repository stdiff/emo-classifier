import json
from pathlib import Path
from importlib import resources
from typing import Optional, Iterator, BinaryIO

import numpy as np
import pandas as pd
from emo_classifier.classifiers.metrics import stats_roc_auc

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
import pytorch_lightning as pl


from emo_classifier.api import Comment, Prediction
from emo_classifier.metrics import Thresholds
from emo_classifier.model import Model
from emo_classifier.emotion import load_emotions
from emo_classifier.classifiers.text import SpacyEnglishTokenizer, load_vocab


with_lemmatization = False
remove_stopwords = True
padding_token = "<pad>"
unknown_token = "<unk>"


class EmbeddingBagModule(pl.LightningModule):
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_labels = len(load_emotions())

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.linear_middle = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.n_labels, bias=False)

        self.weights = torch.tensor([2] * self.n_labels)
        self.loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch size, text length)
        :return: Tensor of shape (batch size, # labels)
        """
        x = self.embedding(x).mean(dim=1)
        x = self.linear_middle(x)
        return self.linear(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, Y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y_true = batch
        Y_prob = self(X)
        return Y_true, Y_prob

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        Y_true = torch.vstack([Y_true for Y_true, _ in outputs])
        Y_hat = torch.vstack([Y_hat for _, Y_hat in outputs])
        print(Y_true.shape, Y_true.dtype, Y_hat.shape, Y_hat.dtype)
        loss = self.loss(Y_hat, Y_true)
        roc_stats = stats_roc_auc(Y_true.numpy(), Y_hat.numpy())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_roc_auc", roc_stats.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> tuple[Tensor, Tensor]:
        X, Y_true = batch
        Y_Prob = torch.softmax(self(X), dim=1)
        return Y_true, Y_Prob

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class EmbeddingBagClassifier(Model):
    artifact_file_name = "embedding_bag.pt"

    def __init__(self, embedding_bag_model: EmbeddingBagModule, vocab: Optional[Vocab] = None):
        self.tokenizer = SpacyEnglishTokenizer(with_lemmatization, remove_stopwords)
        self.vocab = vocab or load_vocab()
        self.vocab.set_default_index(self.vocab[unknown_token])
        self.model = embedding_bag_model
        self.padding_index = self.vocab[padding_token]

        self._thresholds: Optional[Thresholds] = None
        self._s_thresholds: Optional[pd.Series] = None
        self._dict_thresholds: Optional[dict[str, float]] = None

    def texts2tensor(self, texts: Iterator[str]) -> torch.Tensor:
        sequences_of_indices = [torch.tensor(self.vocab(self.tokenizer(text))) for text in texts]
        return pad_sequence(sequences_of_indices, batch_first=True, padding_value=self.padding_index)

    @classmethod
    def load_artifact_file(cls, fp: BinaryIO) -> "EmbeddingBagClassifier":
        with resources.open_binary("emo_classifier.artifact", "hyperparameter.json") as fo:
            hyperparameter = json.load(fo)

        model = EmbeddingBagModule(**hyperparameter)
        model.load_state_dict(torch.load(fp))
        return cls(model)

    def save_artifact_file(self, path: Path):
        hyperparameter_json_path = path.parent / "hyperparameter.json"
        hyperparameter = {"vocab_size": self.model.vocab_size, "embedding_dim": self.model.embedding_dim}
        with hyperparameter_json_path.open("w") as fo:
            json.dump(hyperparameter, fo)
        torch.save(self.model.state_dict(), path)

    @property
    def thresholds(self) -> Optional[Thresholds]:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds: Thresholds):
        self._thresholds = thresholds
        self._s_thresholds = thresholds.as_series()[self.emotions]
        self._dict_thresholds = thresholds.as_dict()

    def predict(self, comment: Comment) -> Prediction:
        y = self.predict_proba([comment.text])[0, :]
        emotions = [emotion for i, emotion in enumerate(self.emotions) if y[i] > self._dict_thresholds.get(emotion)]
        return Prediction(id=comment.id, labels=emotions)

    def predict_proba(self, texts) -> np.ndarray:
        X = self.texts2tensor(texts)
        if X.shape[1] == 0:
            return np.zeros((X.shape[0], len(self.emotions)))

        self.model.eval()
        with torch.no_grad():
            y = torch.softmax(self.model(X), dim=1)
        return y.detach().numpy()

    def predict_proba_in_batch(self, texts, batch_size: int = 256):
        from itertools import islice

        text_iterator = iter(texts)
        ys = []
        while batch := list(islice(text_iterator, batch_size)):
            ys.append(self.predict_proba(batch))

        return np.vstack(ys)
