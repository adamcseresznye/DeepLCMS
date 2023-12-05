from typing import Optional, Tuple

import colab_utils
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from timm import create_model
from torchmetrics import Accuracy
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class MetricsCallback(Callback):
    def __init__(self):
        """
        Callback to capture metrics during the validation phase.

        The captured metrics are stored in the `metrics` list.

        Example:
            ```python
            callback = MetricsCallback()
            trainer = Trainer(callbacks=[callback])
            trainer.fit(model, train_dataloader, val_dataloader)
            print(callback.metrics)
            ```

        """
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called when the validation epoch ends.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The PyTorch Lightning module (model).
        """
        self.metrics.append(trainer.logged_metrics)


class PretrainedModelEvaluator(pl.LightningModule):
    """
    A PyTorch Lightning module for evaluating the performance of pretrained models.

    Args:
        pretrained_model (str): The name of the pretrained model to use.

    Raises:
        ValueError: If no linear layer is found in the pretrained model.

    Attributes:
        model (nn.Module): The pretrained model.
    """

    def __init__(self, pretrained_model: str):
        super().__init__()

        # Use the provided pretrained model
        self.model = create_model(pretrained_model, pretrained=True, num_classes=1)
        # Initialize attributes to store performance metrics
        self.train_loss = []
        self.train_acc = []
        self.train_f1 = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []

        # Freeze all layers except for the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Find the last linear layer dynamically
        last_linear_layer = self.get_last_linear_layer(self.model)

        if last_linear_layer is None:
            raise ValueError("No linear layer found in the pretrained model.")

        # Adjust the last layer dynamically
        last_layer_input_size = last_linear_layer.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=last_layer_input_size,
                out_features=int(last_layer_input_size / 4),
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(
                in_features=int(last_layer_input_size / 4),
                out_features=int(last_layer_input_size / 8),
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=int(last_layer_input_size / 8), out_features=1, bias=True
            ),
        )

    def get_last_linear_layer(self, model: nn.Module) -> Optional[nn.Linear]:
        """
        Recursively find the last linear layer in a given model.

        Args:
            model (nn.Module): The model to search for the last linear layer.

        Returns:
            Optional[nn.Linear]: The last linear layer if found, else None.
        """
        last_linear_layer = None

        for layer in reversed(list(model.children())):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break
            elif list(
                layer.children()
            ):  # If the layer has children, search recursively
                last_linear_layer = self.get_last_linear_layer(layer)
                if last_linear_layer is not None:
                    break

        return last_linear_layer

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

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss_fn = nn.BCELoss()

        y_pred_logits = self(x).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)
        loss = loss_fn(y_pred, y.float())

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics

        # Calculate Accuracy
        y_pred_class = torch.round(y_pred)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate F1
        metric_f1 = BinaryF1Score().to(y.device)
        f1 = metric_f1(y_pred_class, y)
        self.log(
            "train_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate Precision
        metric_precision = BinaryPrecision().to(y.device)
        precision = metric_precision(y_pred_class, y)
        self.log(
            "train_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Calculate Recall
        metric_f1 = BinaryRecall().to(y.device)
        recall = metric_f1(y_pred_class, y)
        self.log(
            "train_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss_fn = nn.BCELoss()

        y_pred_logits = self(x).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)
        loss = loss_fn(y_pred, y.float())
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics

        # Calculate Accuracy
        y_pred_class = torch.round(y_pred)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate F1
        metric_f1 = BinaryF1Score().to(y.device)
        f1 = metric_f1(y_pred_class, y)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Calculate Precision
        metric_precision = BinaryPrecision().to(y.device)
        precision = metric_precision(y_pred_class, y)
        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Calculate Recall
        metric_f1 = BinaryRecall().to(y.device)
        recall = metric_f1(y_pred_class, y)
        self.log(
            "val_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers for PyTorch Lightning.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: Optimizers and schedulers.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            weight_decay=2e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0
        )
        return [optimizer], [scheduler]

    # Add methods to access the performance metrics after training
    def get_train_performance(self):
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "train_f1": self.train_f1,
        }

    def get_val_performance(self):
        return {
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_f1": self.val_f1,
        }


def show_architecture(model: nn.Module):
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


class Resnet_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model("resnet50d.a3_in1k", pretrained=True, num_classes=1)

        # Freeze all layers except for the last one
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.257247),
            torch.nn.Linear(in_features=512, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss_fn = nn.BCELoss()

        y_pred_logits = self(x).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)
        loss = loss_fn(y_pred, y.float())

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics

        # Calculate Accuracy
        y_pred_class = torch.round(y_pred)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        # Calculate F1
        metric_f1 = BinaryF1Score().to(y.device)
        f1 = metric_f1(y_pred_class, y)
        self.log(
            "train_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        # Calculate Precision
        metric_precision = BinaryPrecision().to(y.device)
        precision = metric_precision(y_pred_class, y)
        self.log(
            "train_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # Calculate Recall
        metric_f1 = BinaryRecall().to(y.device)
        recall = metric_f1(y_pred_class, y)
        self.log(
            "train_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss_fn = nn.BCELoss()

        y_pred_logits = self(x).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)
        loss = loss_fn(y_pred, y.float())
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics

        # Calculate Accuracy
        y_pred_class = torch.round(y_pred)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate F1
        metric_f1 = BinaryF1Score().to(y.device)
        f1 = metric_f1(y_pred_class, y)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate Precision
        metric_precision = BinaryPrecision().to(y.device)
        precision = metric_precision(y_pred_class, y)
        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Calculate Recall
        metric_f1 = BinaryRecall().to(y.device)
        recall = metric_f1(y_pred_class, y)
        self.log(
            "val_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list):
            # Assuming the first element in the list is the input tensor
            input_tensor = batch[0]
            return self(input_tensor)
        else:
            # If batch is already a tensor, proceed as usual
            print("Input Shape:", batch.shape)
            return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.002731,
            weight_decay=2e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0
        )
        return [optimizer], [scheduler]
