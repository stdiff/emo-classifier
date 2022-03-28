import json
from datetime import datetime

from sklearn.metrics import log_loss

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from emo_classifier.classifiers.text import save_vocab
from emo_classifier.classifiers.embedding_bag import (
    EmbeddingBagModule,
    with_lemmatization,
    remove_stopwords,
    EmbeddingBagClassifier,
)
from emo_classifier.metrics import TrainingMetrics
from emo_classifier.classifiers.metrics import compute_probabilities, stats_roc_auc
from training import TrainerBase, LocalPaths
from training.pl_logger import SimpleLogger
from training.data_module import GoEmotionsDataModule
from training.utils_for_sagemaker import copy_artifacts_for_outputs_if_on_sagemaker

local_paths = LocalPaths()


class EmbeddingBagTrainer(TrainerBase):
    def __init__(self, embedding_dim: int = 64, max_epoch: int = 3, patience: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_epoch = max_epoch
        self.patience = patience

        if local_paths.on_sagemaker:
            self.log_dir = local_paths.sm_output_data_dir / "training/logs"
        else:
            self.log_dir = local_paths.project_root / "training/logs"

    def fit(self, data_module: GoEmotionsDataModule):
        torch.manual_seed(1)

        data_module.setup("fit")
        save_vocab(data_module.vocab)
        print(f"Vocab is fitted and saved. vocab_size = {len(data_module.vocab)}")
        model = EmbeddingBagModule(vocab_size=len(data_module.vocab), embedding_dim=self.embedding_dim)

        early_stopping = EarlyStopping(monitor="val_roc_auc", patience=self.patience)
        model_checkpoint = ModelCheckpoint(
            monitor="val_roc_auc",
            dirpath="tmp_checkpoint",
            filename="emo-classifier-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        simple_logger = SimpleLogger()
        tensorboard_logger = TensorBoardLogger(save_dir=str(self.log_dir))
        trainer = Trainer(
            max_epochs=self.max_epoch,
            callbacks=[early_stopping, model_checkpoint],
            logger=[simple_logger, tensorboard_logger],
            deterministic=True,
        )
        trainer.fit(model, data_module)
        self.logger.info(f"Model fitted")

        self.classifier = EmbeddingBagClassifier(model)
        y_true, y_prob = compute_probabilities(self.classifier.model, data_module.val_dataloader())

        print("input:", y_prob.dtype, y_prob)
        print("target:", y_true.dtype, y_true)

        self.training_metrics = TrainingMetrics(
            log_loss=self.classifier.model.loss(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item(),
            auc_roc=stats_roc_auc(y_true, y_prob).as_dict(),
            best_params=json.dumps(
                dict(embedding_dim=self.embedding_dim, max_epoch=self.max_epoch, patience=self.patience)
            ),
            timestamp=datetime.now().astimezone().isoformat(),  ## ISO format with timezone
        )


def start_train_embedding_bag_model(embedding_dim: int = 64, max_epoch: int = 3, patience: int = 3):
    data_module = GoEmotionsDataModule(with_lemmatization=with_lemmatization, remove_stopwords=remove_stopwords)
    trainer = EmbeddingBagTrainer(embedding_dim=embedding_dim, max_epoch=max_epoch, patience=patience)
    trainer.fit(data_module)
    trainer.save_model()
    copy_artifacts_for_outputs_if_on_sagemaker()


if __name__ == "__main__":
    start_train_embedding_bag_model(embedding_dim=64, max_epoch=20, patience=3)
