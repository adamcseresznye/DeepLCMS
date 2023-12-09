from typing import Optional, Tuple

import colab_functions
import colab_utils
import pandas as pd
import prepare_data
import timm
import torch
import torchinfo
import torchmetrics
from lightning.pytorch import LightningModule


class PretrainedModel(LightningModule):
    def __init__(
        self, pretrained_model_name: str, learning_rate: float, freeze: bool = True
    ):
        """
        LightningModule for fine-tuning a pretrained model on a binary classification task.

        Args:
            pretrained_model_name (str): Name of the pretrained model architecture from timm.
            learning_rate (float): Learning rate for the optimizer.
            freeze (bool, optional): If True, freeze all layers except for the final layer.
                Defaults to True.

        Attributes:
            pretrained_model_name (str): Name of the pretrained model architecture.
            model (torch.nn.Module): The pretrained model.
            learning_rate (float): Learning rate for the optimizer.
            loss_fn (torch.nn.Module): Binary cross-entropy with logits loss.
            accuracy (torchmetrics.classification.BinaryAccuracy): Binary accuracy metric.
            f1 (torchmetrics.classification.BinaryF1Score): Binary F1 score metric.
            precision (torchmetrics.classification.BinaryPrecision): Binary precision metric.
            recall (torchmetrics.classification.BinaryRecall): Binary recall metric.

        Note:
            All layers of the pretrained model are frozen except for the final layer if `freeze` is True.

        """
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.model = timm.create_model(
            pretrained_model_name, pretrained=True, num_classes=1
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.precision = torchmetrics.classification.BinaryPrecision()
        self.recall = torchmetrics.classification.BinaryRecall()

        if freeze:
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            # Get the last layer
            last_layer = None
            for child in self.model.named_children():
                last_layer = child

            # Unfreeze the last layer
            if last_layer is not None:
                for param in last_layer[1].parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.model(x)
        return x

    def common_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step for both training and validation.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch (x, y).
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing loss, predicted logits, and target tensor.

        """
        x, y = batch
        y_pred_logits = self(x).squeeze()
        loss = self.loss_fn(y_pred_logits, y.float())
        return loss, y_pred_logits, y

    def log_metrics(
        self,
        prefix: str,
        accuracy: torch.Tensor,
        f1: torch.Tensor,
        precision: torch.Tensor,
        recall: torch.Tensor,
    ) -> None:
        """
        Log metrics during training or validation.

        Args:
            prefix (str): Prefix for metric names.
            accuracy (torch.Tensor): Binary accuracy.
            f1 (torch.Tensor): Binary F1 score.
            precision (torch.Tensor): Binary precision.
            recall (torch.Tensor): Binary recall.

        """
        self.log_dict(
            {
                f"{prefix}_accuracy": accuracy,
                f"{prefix}_f1": f1,
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch (x, y).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss for the training step.

        """
        loss, y_pred_logits, y = self.common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred_logits, y)
        f1 = self.f1(y_pred_logits, y)
        precision = self.precision(y_pred_logits, y)
        recall = self.recall(y_pred_logits, y)

        self.log_metrics("train", accuracy, f1, precision, recall)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch (x, y).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss for the validation step.

        """
        loss, y_pred_logits, y = self.common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred_logits, y)
        f1 = self.f1(y_pred_logits, y)
        precision = self.precision(y_pred_logits, y)
        recall = self.recall(y_pred_logits, y)

        self.log_metrics("val", accuracy, f1, precision, recall)
        return loss

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Predict step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch (x, y).
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            torch.Tensor: Model predictions.

        """
        if isinstance(batch, list):
            input_tensor = batch[0]
            return self(input_tensor)
        else:
            print("Input Shape:", batch.shape)
            return self(batch)

    def configure_optimizers(self) -> Tuple:
        """
        Configure optimizers and schedulers.

        Returns:
            Tuple[List[Optimizer], List[_LRScheduler]]: Optimizers and schedulers.

        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=2e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0
        )
        return [optimizer], [scheduler]


def show_architecture(model: torch.nn.Module):
    """
    Display a summary of the model architecture using torchinfo.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        str: A summary of the model architecture.
    """
    return torchinfo.summary(
        model=model,
        input_size=(32, 3, 384, 384),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
