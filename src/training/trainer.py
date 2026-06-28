"""Un bucle de entrenamiento de PyTorch transparente para uso educativo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss


class MultiTaskTrainer:
    """Entrena, valida y guarda checkpoints de un modelo multitarea de PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: MultiTaskLoss,
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        """Configura el entrenador con el modelo, el optimizador y la función de pérdida.

        Args:
            model: Modelo multitarea de PyTorch a entrenar.
            optimizer: Optimizador usado para actualizar los parámetros del modelo.
            loss_function: Función de pérdida combinada de género y edad.
            device: Dispositivo (CPU/GPU) donde se ejecutan el modelo y los datos.
            checkpoint_path: Ruta del archivo donde se guarda el mejor checkpoint.
            checkpoint_metadata: Metadatos adicionales incluidos en cada checkpoint.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = checkpoint_metadata

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> tuple[list[dict[str, float]], float]:
        """Ejecuta todas las épocas y guarda el checkpoint con la menor pérdida de validación.

        Args:
            train_loader: DataLoader con los lotes de entrenamiento.
            val_loader: DataLoader con los lotes de validación.
            epochs: Número de épocas a ejecutar.

        Returns:
            Tupla con el historial de métricas por época y el tiempo total de
            entrenamiento en segundos.
        """

        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_losses = self._run_epoch(train_loader, training=True)
            val_losses = self._run_epoch(val_loader, training=False)
            epoch_seconds = time.perf_counter() - epoch_start

            row = {
                "epoch": float(epoch),
                "train_total_loss": train_losses["total_loss"],
                "train_gender_loss": train_losses["gender_loss"],
                "train_age_loss": train_losses["age_loss"],
                "val_total_loss": val_losses["total_loss"],
                "val_gender_loss": val_losses["gender_loss"],
                "val_age_loss": val_losses["age_loss"],
                "epoch_seconds": epoch_seconds,
            }
            history.append(row)

            print(
                f"  epoch {epoch:02d}/{epochs:02d} "
                f"train={row['train_total_loss']:.4f} "
                f"val={row['val_total_loss']:.4f} "
                f"({epoch_seconds:.1f}s)"
            )

            if row["val_total_loss"] < best_val_loss:
                best_val_loss = row["val_total_loss"]
                self._save_checkpoint(epoch=epoch, val_loss=best_val_loss)

        training_seconds = time.perf_counter() - start_time
        return history, training_seconds

    def load_best_checkpoint(self) -> dict[str, Any]:
        """Restaura el modelo seleccionado mediante la pérdida de validación.

        Returns:
            Diccionario del checkpoint cargado, incluyendo metadatos y el estado
            del modelo.
        """

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        """Ejecuta una época completa de entrenamiento o validación.

        Args:
            loader: DataLoader con los lotes de imágenes y etiquetas de género y edad.
            training: Si es True, actualiza los pesos del modelo; si es False, solo evalúa.

        Returns:
            Diccionario con el promedio de pérdida total, de género y de edad
            sobre todas las muestras procesadas.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        totals = {"total_loss": 0.0, "gender_loss": 0.0, "age_loss": 0.0}
        sample_count = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, gender_targets, age_targets in loader:
                images = images.to(self.device)
                gender_targets = gender_targets.to(self.device)
                age_targets = age_targets.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                gender_logits, age_predictions = self.model(images)
                losses = self.loss_function(
                    gender_logits,
                    age_predictions,
                    gender_targets,
                    age_targets,
                )

                if training:
                    losses.total.backward()
                    self.optimizer.step()

                batch_size = images.size(0)
                sample_count += batch_size
                totals["total_loss"] += losses.total.item() * batch_size
                totals["gender_loss"] += losses.gender.item() * batch_size
                totals["age_loss"] += losses.age.item() * batch_size

        if sample_count == 0:
            raise RuntimeError("El DataLoader no contiene muestras.")
        return {name: value / sample_count for name, value in totals.items()}

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Guarda el estado del modelo y los metadatos del experimento en disco.

        Args:
            epoch: Número de época en la que se generó el checkpoint.
            val_loss: Pérdida de validación asociada a este checkpoint.
        """
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            **self.checkpoint_metadata,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
