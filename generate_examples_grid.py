"""Generate a qualitative grid of E5 (ResNet18 fine-tuning) predictions on the test split."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet
from src.utils import resolve_device

GENDER_LABELS = {0: "Masculino", 1: "Femenino"}
AGE_ERROR_THRESHOLD = 5.0
MAX_EXAMPLES = 12
GRID_COLS = 4

MODEL_FACTORIES = {
    "cnn": MultiTaskCNN,
    "mlp": MultiTaskMLP,
    "resnet_frozen": MultiTaskResNet,
    "resnet_finetuning": MultiTaskResNet,
}


def _build_model_from_checkpoint(checkpoint: dict) -> torch.nn.Module:
    model_name = checkpoint["model_name"]
    factory = MODEL_FACTORIES.get(model_name)
    if factory is None:
        raise ValueError(f"model_name desconocido en el checkpoint: {model_name}")

    model = factory(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _select_example_indices(
    ages: list[float],
    gender_true: list[int],
    gender_pred: list[int],
    num_examples: int,
) -> list[int]:
    """Pick a varied subset: a young sample, an elderly sample, a gender error, plus filler."""

    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index not in selected:
            selected.append(index)

    add(next((i for i, age in enumerate(ages) if age < 13.0), None))
    add(next((i for i, age in enumerate(ages) if age >= 60.0), None))
    add(next((i for i, (gt, gp) in enumerate(zip(gender_true, gender_pred)) if gt != gp), None))

    remaining = [i for i in range(len(ages)) if i not in selected]
    slots_left = max(1, num_examples - len(selected))
    step = max(1, len(remaining) // slots_left)
    for index in remaining[::step]:
        if len(selected) >= num_examples:
            break
        add(index)

    return selected[:num_examples]


def main() -> None:
    config = AppConfig.from_env()
    checkpoint_path = config.checkpoints_dir / "resnet_finetuning_base" / "best_model.pt"

    if not checkpoint_path.exists():
        print(
            "No se encontro el checkpoint de E5 en: "
            f"{checkpoint_path}. Ejecuta "
            "'python main.py --experiment resnet_finetuning_base' antes de generar esta figura."
        )
        return

    device = resolve_device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    checkpoint_image_size = checkpoint.get("image_size")
    if checkpoint_image_size is not None and checkpoint_image_size != config.image_size:
        print(
            f"Aviso: el checkpoint se entreno con image_size={checkpoint_image_size}, "
            f"pero la configuracion actual usa IMAGE_SIZE={config.image_size}."
        )

    model = _build_model_from_checkpoint(checkpoint).to(device)

    data_module = UTKFaceDataModule(config, use_augmentation=False)
    data_module.setup()
    test_dataset = data_module.test_dataset
    test_loader = data_module.test_dataloader()

    gender_true: list[int] = []
    gender_pred: list[int] = []
    age_true: list[float] = []
    age_pred: list[float] = []

    with torch.inference_mode():
        for images, batch_gender, batch_age in test_loader:
            gender_logits, batch_age_pred = model(images.to(device))
            gender_pred.extend(gender_logits.argmax(dim=1).cpu().tolist())
            gender_true.extend(batch_gender.tolist())
            age_true.extend(batch_age.tolist())
            age_pred.extend(batch_age_pred.cpu().tolist())

    num_examples = min(MAX_EXAMPLES, len(age_true))
    indices = _select_example_indices(age_true, gender_true, gender_pred, num_examples)

    grid_cols = min(GRID_COLS, len(indices))
    grid_rows = math.ceil(len(indices) / grid_cols)

    output_dir = config.plots_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_grid.png"

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(4 * grid_cols, 4.5 * grid_rows),
        squeeze=False,
    )
    fig.suptitle("Predicciones E5 (ResNet18 fine-tuning) - test split", fontsize=14)

    axes_list = list(axes.flat)
    for ax, index in zip(axes_list, indices):
        record = test_dataset.records[index]
        with Image.open(record.path) as image_file:
            image = image_file.convert("RGB")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

        gender_ok = gender_true[index] == gender_pred[index]
        age_ok = abs(age_true[index] - age_pred[index]) <= AGE_ERROR_THRESHOLD

        gender_line = (
            f"Genero real: {GENDER_LABELS[gender_true[index]]} | "
            f"pred: {GENDER_LABELS[gender_pred[index]]}"
        )
        age_line = f"Edad real: {age_true[index]:.0f} | pred: {age_pred[index]:.1f}"

        ax.set_title(gender_line, color="#2A9D2A" if gender_ok else "#D62728", fontsize=10)
        ax.text(
            0.5,
            -0.08,
            age_line,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#2A9D2A" if age_ok else "#D62728",
        )

        border_color = "#2A9D2A" if (gender_ok and age_ok) else "#D62728"
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)

    for ax in axes_list[len(indices):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Figura generada en: {output_path}")
    print(f"Ejemplos incluidos: {len(indices)} de {len(test_dataset)} muestras de test.")


if __name__ == "__main__":
    main()
