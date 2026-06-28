"""Detecta y recorta el rostro más grande de una imagen nueva."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class FaceDetection:
    """El recorte del rostro seleccionado y su caja delimitadora en la imagen original."""

    crop: Image.Image
    box: tuple[int, int, int, int]


class FaceDetector:
    """Usa el clasificador en cascada Haar de OpenCV para un despliegue educativo pequeño."""

    def __init__(self, cascade_path: str | Path | None = None) -> None:
        """Carga el clasificador en cascada Haar para detección de rostros frontales.

        Args:
            cascade_path: Ruta al archivo XML del clasificador en cascada. Si es
                None, usa el cascade frontal por defecto incluido en OpenCV.

        Raises:
            RuntimeError: Si el clasificador no se pudo cargar desde `cascade_path`.
        """
        if cascade_path is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade_path = Path(cascade_path)
        self.detector = cv2.CascadeClassifier(str(self.cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"No se pudo cargar el detector facial: {self.cascade_path}")

    def detect_largest(self, image: Image.Image) -> FaceDetection | None:
        """Detecta el rostro frontal más grande de la imagen.

        Args:
            image: Imagen de entrada en la que buscar rostros.

        Returns:
            `FaceDetection` con el recorte y la caja del rostro más grande, o
            None si no se detectó ningún rostro.
        """

        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
        box = (int(x), int(y), int(x + width), int(y + height))
        return FaceDetection(crop=rgb_image.crop(box), box=box)

    @staticmethod
    def draw_box(image: Image.Image, detection: FaceDetection) -> Image.Image:
        """Crea una copia de la imagen con el rostro seleccionado resaltado.

        Args:
            image: Imagen original sobre la que dibujar.
            detection: Detección cuya caja se va a resaltar.

        Returns:
            Copia de la imagen con un rectángulo rojo alrededor del rostro detectado.
        """

        annotated = image.convert("RGB").copy()
        drawer = ImageDraw.Draw(annotated)
        drawer.rectangle(detection.box, outline="red", width=3)
        return annotated
