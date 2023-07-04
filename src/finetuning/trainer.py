from typing import Dict, Optional, List, NamedTuple, Union, Any

import datasets
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import wandb

from torch import nn

from config import MODELS_DIR, TrainerConfig
from utils import get_device
from finetuning.custom_callbacks import CustomWandbCallback

class PredictionOutput(NamedTuple):
    """Prediction output."""

    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]

class ExperimentTrainer:
    """Experiment Trainer."""

    def __init__(
            self,
            data,
            config,
    ) -> None:
        pass

    def train(self) -> None:
        """Train model. We save the model in the MODELS_DIR directory and log the results to wandb."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=f"{self.config.ckpt}_size:{self.dataset['train'].num_rows}",
            group=f"{self.config.dataset}_{self.config.approach}",
            config={
                "ckpt": self.config.ckpt,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "weight_decay": self.config.weight_decay,
                "train_size": self.dataset["train"].num_rows,
                "use_augmented_data": self.config.use_augmented_data,
                "approach": self.config.approach,
                "augmentation_model": self.config.augmentation_model,
            },
        )

        args = TrainingArguments(
            str(
                MODELS_DIR
                / f"{self.config.dataset}_size:{len(self.dataset['train'])}_{self.config.ckpt}"
            ),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=self.config.lr,
            load_best_model_at_end=True,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            metric_for_best_model="eval_loss",
            push_to_hub=False,
            seed=42,
            warmup_ratio=0.2
        )
            

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[CustomWandbCallback()],
        )

        trainer.train()

        print(trainer.evaluate())
        self.trainer = trainer

    def test(
            self,
    ) -> PredictionOutput:
        """Test the trained model on the test set and log to wandb."""
        prediction_output: PredictionOutput = self.trainer.predict(
            self.dataset["test"], metric_key_prefix=""
        )

        metrics, y_pred_logits, y_true = (
            prediction_output.metrics,
            prediction_output.predictions,
            prediction_output.label_ids,
        )

        # Convert logits to probabilities
        softmax = torch.nn.Softmax()
        probs = softmax(torch.Tensor(y_pred_logits)).numpy()
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1

        clf_report = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=self.labels, output_dict=True
        )

        # Add prefix to metrics "test/"
        metrics = {f"test/{k[1:]}": v for k, v in metrics.items()}
        # Log results
        wandb.log(
            metrics,
        )

        df = pd.DataFrame(clf_report)
        df["metric"] = df.index
        table = wandb.Table(data=df)

        wandb.log(
            {
                "classification_report": table,
            }
        )

        return prediction_output

    def predict(self, x: torch.Tensor) -> PredictionOutput:
        """Predict"""
        return self.trainer.predict(x)
