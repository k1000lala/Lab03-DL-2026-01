"""Dataset de PyTorch para imágenes de UTKFace."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.parser import UTKFaceRecord


class UTKFaceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Carga imágenes de rostros y devuelve tensores para ambas tareas de aprendizaje."""

    def __init__(
        self,
        records: Sequence[UTKFaceRecord],
        transform: Callable | None = None,
    ) -> None:
        """Inicializa el dataset a partir de los registros y la transformación a aplicar.

        Args:
            records: Secuencia de `UTKFaceRecord` con la ruta y las etiquetas de cada imagen.
            transform: Transformación opcional aplicada a cada imagen cargada.
        """
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        """Devuelve la cantidad de registros del dataset."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Carga y transforma la imagen en `index`, junto con sus etiquetas.

        Args:
            index: Posición del registro a cargar.

        Returns:
            Tupla con la imagen transformada, el género (long) y la edad (float32).

        Raises:
            TypeError: Si `transform` no convierte la imagen en `torch.Tensor`.
        """
        record = self.records[index]
        with Image.open(record.path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError("La transformacion debe convertir la imagen a torch.Tensor.")

        gender = torch.tensor(record.gender, dtype=torch.long)
        age = torch.tensor(record.age, dtype=torch.float32)
        return image, gender, age
