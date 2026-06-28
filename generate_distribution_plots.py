"""Generate UTKFace dataset distribution plots from filename labels.

Reads age/gender/race straight from filenames, without instantiating any
PyTorch dataset, so it can run in environments that only have matplotlib.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import AppConfig

AGE_RANGES = (
    ("0-12", 0.0, 13.0),
    ("13-19", 13.0, 20.0),
    ("20-39", 20.0, 40.0),
    ("40-59", 40.0, 60.0),
    ("60+", 60.0, float("inf")),
)

GENDER_LABELS = {0: "Masculino", 1: "Femenino"}

RACE_LABELS = {
    0: "Blanco",
    1: "Negro",
    2: "Asiatico",
    3: "Indio",
    4: "Otros",
}


def _load_parser_module():
    """Import src/data/parser.py directly, bypassing src/data/__init__.py (which imports torch)."""
    parser_path = Path(__file__).resolve().parent / "src" / "data" / "parser.py"
    spec = importlib.util.spec_from_file_location("utkface_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _age_range_label(age: float) -> str:
    for label, lower, upper in AGE_RANGES:
        if lower <= age < upper:
            return label
    return AGE_RANGES[-1][0]


def _plot_age_histogram(ages: list[float], output_path: Path) -> None:
    max_age = max(ages) if ages else 0.0
    bins = list(range(0, int(max_age) + 6, 5))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ages, bins=bins, color="#4C72B0", edgecolor="black")
    ax.set_title("Distribucion de edades - UTKFace")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Cantidad de imagenes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        ax.annotate(
            str(int(bar.get_height())),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )


def _plot_gender_distribution(genders: list[int], output_path: Path) -> dict[int, int]:
    counts = {key: genders.count(key) for key in GENDER_LABELS}
    labels = [GENDER_LABELS[key] for key in sorted(GENDER_LABELS)]
    values = [counts[key] for key in sorted(GENDER_LABELS)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    _annotate_bars(ax, bars)
    ax.set_title("Distribucion por genero - UTKFace")
    ax.set_xlabel("Genero")
    ax.set_ylabel("Cantidad de imagenes")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def _plot_age_range_distribution(age_range_counts: dict[str, int], output_path: Path) -> None:
    labels = [label for label, _, _ in AGE_RANGES]
    values = [age_range_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#55A868")
    _annotate_bars(ax, bars)
    ax.set_title("Distribucion por rango etario - UTKFace")
    ax.set_xlabel("Rango etario")
    ax.set_ylabel("Cantidad de imagenes")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_race_distribution(races: list[int], output_path: Path) -> None:
    keys = sorted(RACE_LABELS)
    counts = {key: races.count(key) for key in keys}
    labels = [RACE_LABELS[key] for key in keys]
    values = [counts[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#C44E52")
    _annotate_bars(ax, bars)
    ax.set_title("Distribucion por etnia - UTKFace")
    ax.set_xlabel("Etnia")
    ax.set_ylabel("Cantidad de imagenes")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = AppConfig.from_env()
    random.seed(config.seed)

    if not config.dataset_dir.exists():
        raise FileNotFoundError(
            f"No existe UTKFACE_DIR: {config.dataset_dir}. "
            "Configure la ruta en el archivo .env."
        )

    parser_module = _load_parser_module()
    parser = parser_module.UTKFaceFilenameParser()

    ages: list[float] = []
    genders: list[int] = []
    races: list[int] = []
    age_range_counts = {label: 0 for label, _, _ in AGE_RANGES}
    discarded = 0

    for path in sorted(config.dataset_dir.glob("*.jpg")):
        try:
            record = parser.parse(path)
        except ValueError:
            discarded += 1
            continue
        ages.append(record.age)
        genders.append(record.gender)
        age_range_counts[_age_range_label(record.age)] += 1
        if record.race is not None:
            races.append(record.race)

    output_dir = config.plots_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_age_histogram(ages, output_dir / "age_histogram.png")
    gender_counts = _plot_gender_distribution(genders, output_dir / "gender_distribution.png")
    _plot_age_range_distribution(age_range_counts, output_dir / "age_range_distribution.png")
    _plot_race_distribution(races, output_dir / "race_distribution.png")

    print(f"Imagenes validas: {len(ages)}")
    print(f"Imagenes descartadas: {discarded}")
    print("Conteo por genero:")
    for key in sorted(GENDER_LABELS):
        print(f"  {GENDER_LABELS[key]} ({key}): {gender_counts[key]}")
    print("Conteo por rango etario:")
    for label, _, _ in AGE_RANGES:
        print(f"  {label}: {age_range_counts[label]}")


if __name__ == "__main__":
    main()
