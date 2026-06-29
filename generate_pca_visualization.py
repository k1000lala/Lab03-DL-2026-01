"""Genera visualizaciones de PCA reutilizando el mismo preprocesamiento que el baseline E1.

Carga una muestra del split de entrenamiento con UTKFaceDataModule (la misma
clase y semilla que usa ClassicalBaseline en experiment_runner.py), ajusta PCA
sobre los pixeles aplanados y guarda dos figuras de diagnostico. No reentrena
ningun modelo ni modifica artefactos existentes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule

# Vectorizar todo el split de entrenamiento (cada imagen son 3*image_size**2
# floats) seria demasiado pesado solo para una visualizacion; se usa una
# muestra aleatoria reproducible del mismo split que usa E1.
SAMPLE_SIZE = 2000
VARIANCE_COMPONENTS = 200
MILESTONE_COMPONENTS = (50, 100, 200)
GENDER_LABELS = {0: "Masculino", 1: "Femenino"}
GENDER_COLORS = {0: "#4C72B0", 1: "#DD8452"}


def _sample_train_arrays(config: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    """Carga una muestra del split de entrenamiento de E1, ya vectorizada.

    Usa UTKFaceDataModule con use_augmentation=False, igual que
    ClassicalBaseline en experiment_runner.py, para que el preprocesamiento
    (resize, normalizacion ImageNet, aplanado) sea idéntico al de E1.

    Args:
        config: Configuración de la aplicación (rutas, semilla, image_size).

    Returns:
        Tupla con la matriz de imágenes aplanadas (forma [N, 3*image_size**2])
        y el arreglo de etiquetas de género (forma [N]).
    """
    data_module = UTKFaceDataModule(config, use_augmentation=False)
    data_module.setup()
    train_dataset = data_module.train_dataset

    sample_size = min(SAMPLE_SIZE, len(train_dataset))
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:sample_size].tolist()
    subset = Subset(train_dataset, indices)

    loader = DataLoader(subset, batch_size=64, shuffle=False)
    images: list[torch.Tensor] = []
    genders: list[torch.Tensor] = []
    for batch_images, batch_gender, _batch_age in loader:
        images.append(batch_images)
        genders.append(batch_gender)

    flattened = torch.cat(images).reshape(sample_size, -1).numpy()
    gender_array = torch.cat(genders).numpy()
    return flattened, gender_array


def _plot_pca_scatter(features: np.ndarray, genders: np.ndarray, output_path: Path) -> None:
    """Proyecta las imágenes en 2D con PCA y grafica un scatter coloreado por género.

    Args:
        features: Matriz de imágenes aplanadas, forma [N, D].
        genders: Etiquetas de género asociadas a cada fila de `features`.
        output_path: Ruta donde se guarda el PNG generado.
    """
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    for gender_value, label in GENDER_LABELS.items():
        mask = genders == gender_value
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=14,
            alpha=0.6,
            color=GENDER_COLORS[gender_value],
            label=label,
        )
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title("Proyección PCA del espacio de rostros por género")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_explained_variance(features: np.ndarray, output_path: Path) -> dict[int, float]:
    """Ajusta PCA con muchos componentes y grafica la varianza explicada acumulada.

    Args:
        features: Matriz de imágenes aplanadas, forma [N, D].
        output_path: Ruta donde se guarda el PNG generado.

    Returns:
        Diccionario con la varianza explicada acumulada (fracción, no %) en
        cada uno de los hitos de `MILESTONE_COMPONENTS` que caben en los datos.
    """
    n_components = min(VARIANCE_COMPONENTS, *features.shape)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_components + 1), cumulative, color="#4C72B0")

    milestones: dict[int, float] = {}
    for milestone in MILESTONE_COMPONENTS:
        if milestone > n_components:
            continue
        value = float(cumulative[milestone - 1])
        milestones[milestone] = value
        ax.axvline(milestone, color="gray", linestyle="--", alpha=0.6)
        ax.scatter([milestone], [value], color="#C44E52", zorder=5)
        ax.annotate(
            f"{value * 100:.1f}%",
            xy=(milestone, value),
            xytext=(5, -10),
            textcoords="offset points",
        )

    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("Varianza explicada acumulada por PCA")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return milestones


def main() -> None:
    """Genera y guarda las dos figuras de PCA a partir de una muestra del split de entrenamiento."""
    config = AppConfig.from_env()
    if not config.dataset_dir.exists():
        raise FileNotFoundError(
            f"No existe UTKFACE_DIR: {config.dataset_dir}. "
            "Configure la ruta en el archivo .env."
        )

    features, genders = _sample_train_arrays(config)
    print(f"Muestra cargada: {features.shape[0]} imagenes, {features.shape[1]} dimensiones.")

    output_dir = config.plots_dir / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_pca_scatter(features, genders, output_dir / "pca_scatter_gender.png")
    milestones = _plot_explained_variance(features, output_dir / "pca_varianza_explicada.png")

    print("Varianza explicada acumulada:")
    for n_components, value in milestones.items():
        print(f"  {n_components} componentes: {value * 100:.2f}%")


if __name__ == "__main__":
    main()
