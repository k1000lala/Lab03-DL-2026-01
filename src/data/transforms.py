"""Transformaciones de imagen compartidas entre entrenamiento e inferencia."""

from __future__ import annotations

from torchvision import transforms


class TransformFactory:
    """Crea pipelines de torchvision deterministas y con aumentación."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @classmethod
    def training(cls, image_size: int, use_augmentation: bool = True):
        """Construye el pipeline de transformaciones para entrenamiento.

        Args:
            image_size: Alto y ancho (en píxeles) al que se redimensionan las imágenes.
            use_augmentation: Si es True, agrega flip horizontal aleatorio y color jitter.

        Returns:
            `transforms.Compose` con el pipeline de transformaciones de entrenamiento.
        """
        operations = [transforms.Resize((image_size, image_size))]
        if use_augmentation:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                    ),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ]
        )
        return transforms.Compose(operations)

    @classmethod
    def evaluation(cls, image_size: int):
        """Construye el pipeline determinista usado en validación, prueba e inferencia.

        Args:
            image_size: Alto y ancho (en píxeles) al que se redimensionan las imágenes.

        Returns:
            `transforms.Compose` sin aleatoriedad (sin aumentación de datos).
        """

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ]
        )

    @classmethod
    def inference(cls, image_size: int):
        """Construye el pipeline de transformaciones usado en inferencia.

        Args:
            image_size: Alto y ancho (en píxeles) al que se redimensionan las imágenes.

        Returns:
            `transforms.Compose` idéntico al de `evaluation`.
        """
        return cls.evaluation(image_size)
