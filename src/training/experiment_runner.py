"""Catálogo de experimentos y orquestación del laboratorio."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.baselines.classical_todo import ClassicalBaseline
from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.data.dataset import UTKFaceDataset
from src.evaluation.metrics import MultiTaskEvaluator, MultiTaskMetrics
from src.evaluation.plots import ResultPlotter
from src.evaluation.reporter import ExperimentResult, ExperimentStatus
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet
from src.training.losses import MultiTaskLoss
from src.training.trainer import MultiTaskTrainer
from src.utils import set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    """Configuración de un experimento base o de una ablación de un solo cambio."""

    strategy_id: str
    strategy_name: str
    name: str
    variant: str
    changed_component: str
    implemented: bool
    model_kind: str
    use_augmentation: bool = True
    dropout: float = 0.4
    lambda_age: float = 0.01
    learning_rate: float = 1e-3
    trainable_blocks: int = 0
    pca_components: int = 100


def build_experiment_catalog(config: AppConfig) -> dict[str, ExperimentSpec]:
    """Devuelve todas las estrategias requeridas y sus estudios de ablación esperados.

    E3 se entrega como ejemplo completo. E1, E2, E4 y E5 son intencionalmente
    visibles pero no implementados, para que los alumnos los completen y reporten.

    Args:
        config: Configuración de la aplicación, usada para derivar `lambda_age`
            base y los `learning_rate` de cada spec.

    Returns:
        Diccionario que mapea el nombre de cada experimento a su `ExperimentSpec`.
    """

    low_lambda = config.lambda_age / 10
    high_lambda = config.lambda_age * 10

    specs = [
        # E1: classical baseline. The estimator itself will use scikit-learn.
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_base", "base", "ninguno",
            True, "classical", pca_components=100,
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_pca_50", "ablacion", "PCA=50 componentes",
            True, "classical", pca_components=50,
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_pca_200", "ablacion", "PCA=200 componentes",
            True, "classical", pca_components=200,
        ),
        # E2: students must implement both the base MLP and its ablations.
        ExperimentSpec(
            "E2",
            "MLP multitarea",
            "mlp_base",
            "base",
            "ninguno",
            True,
            "mlp",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2",
            "MLP multitarea",
            "mlp_no_dropout",
            "ablacion",
            "dropout=0.0",
            True,
            "mlp",
            use_augmentation=True,
            dropout=0.0,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2",
            "MLP multitarea",
            "mlp_lambda_low",
            "ablacion",
            f"lambda_age={low_lambda:g}",
            True,
            "mlp",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=low_lambda,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2",
            "MLP multitarea",
            "mlp_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "mlp",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
        ),
        # E3: complete PyTorch CNN example and one-change-at-a-time ablations.
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_base",
            "base",
            "ninguno",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_augmentation",
            "ablacion",
            "sin aumentacion",
            True,
            "cnn",
            use_augmentation=False,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_dropout",
            "ablacion",
            "dropout=0.0",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.0,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_low",
            "ablacion",
            f"lambda_age={low_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=low_lambda,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
        ),
        # E4: frozen ResNet transfer learning exercises.
        ExperimentSpec(
            "E4",
            "ResNet18 congelada",
            "resnet_frozen_base",
            "base",
            "ninguno",
            True,
            "resnet_frozen",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E4",
            "ResNet18 congelada",
            "resnet_frozen_no_augmentation",
            "ablacion",
            "sin aumentacion",
            True,
            "resnet_frozen",
            use_augmentation=False,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E4",
            "ResNet18 congelada",
            "resnet_frozen_lambda_low",
            "ablacion",
            f"lambda_age={low_lambda:g}",
            True,
            "resnet_frozen",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=low_lambda,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E4",
            "ResNet18 congelada",
            "resnet_frozen_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "resnet_frozen",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
        ),
        # E5: fine-tuning exercises.
        ExperimentSpec(
            "E5",
            "ResNet18 fine-tuning",
            "resnet_finetuning_base",
            "base",
            "ninguno",
            True,
            "resnet_finetuning",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
            trainable_blocks=1,
        ),
        ExperimentSpec(
            "E5",
            "ResNet18 fine-tuning",
            "resnet_finetuning_unfreeze_more",
            "ablacion",
            "mas bloques descongelados",
            True,
            "resnet_finetuning",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
            trainable_blocks=2,
        ),
        ExperimentSpec(
            "E5",
            "ResNet18 fine-tuning",
            "resnet_finetuning_lr_low",
            "ablacion",
            "learning rate menor",
            True,
            "resnet_finetuning",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate / 10,
            trainable_blocks=1,
        ),
        ExperimentSpec(
            "E5",
            "ResNet18 fine-tuning",
            "resnet_finetuning_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "resnet_finetuning",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
            trainable_blocks=1,
        ),
    ]
    return {spec.name: spec for spec in specs}


class ExperimentRunner:
    """Ejecuta los experimentos seleccionados y conserva filas de reporte para cada estrategia."""

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        catalog: dict[str, ExperimentSpec],
    ) -> None:
        """Inicializa el runner con la configuración, el dispositivo y el catálogo de experimentos.

        Args:
            config: Configuración de la aplicación.
            device: Dispositivo (CPU/GPU) donde se entrenan y evalúan los modelos.
            catalog: Diccionario de nombre de experimento a `ExperimentSpec`.
        """
        self.config = config
        self.device = device
        self.catalog = catalog
        self.plotter = ResultPlotter(config.plots_dir)

    def run(self, selected_names: set[str]) -> list[ExperimentResult]:
        """Ejecuta los experimentos seleccionados y genera los gráficos de ablación.

        Los experimentos no implementados o no seleccionados se reportan con su
        estado correspondiente sin ejecutarse.

        Args:
            selected_names: Nombres de los experimentos del catálogo a ejecutar.

        Returns:
            Lista de `ExperimentResult`, uno por cada experimento del catálogo.

        Raises:
            ValueError: Si `selected_names` contiene nombres que no están en el catálogo.
        """
        unknown = selected_names.difference(self.catalog)
        if unknown:
            raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")

        results: list[ExperimentResult] = []
        for spec in self.catalog.values():
            if not spec.implemented:
                results.append(self._not_implemented_result(spec))
            elif spec.name not in selected_names:
                results.append(self._not_executed_result(spec))
            else:
                results.append(self._run_spec(spec))

        for strategy_id in ("E1", "E2", "E3", "E4", "E5"):
            self.plotter.plot_ablation_comparison(results, strategy_id)
        return results

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        """Entrena y evalúa un experimento neuronal (o delega al baseline clásico).

        Args:
            spec: Especificación del experimento a ejecutar.

        Returns:
            `ExperimentResult` con las métricas, o con estado de error si la
            ejecución falla.
        """
        if spec.model_kind == "classical":
            return self._run_classical_spec(spec)

        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            data_module = UTKFaceDataModule(
                self.config,
                use_augmentation=spec.use_augmentation,
            )
            data_module.setup()

            model, model_kwargs = self._build_model(spec)
            model = model.to(self.device)
            optimizer = optim.Adam(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=spec.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            loss_function = MultiTaskLoss(lambda_age=spec.lambda_age)
            checkpoint_path = (
                self.config.checkpoints_dir / spec.name / "best_model.pt"
            )
            trainer = MultiTaskTrainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                device=self.device,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata={
                    "experiment_name": spec.name,
                    "strategy_id": spec.strategy_id,
                    "model_name": spec.model_kind,
                    "model_kwargs": model_kwargs,
                    "image_size": self.config.image_size,
                    "lambda_age": spec.lambda_age,
                },
            )
            history, training_seconds = trainer.fit(
                data_module.train_dataloader(),
                data_module.val_dataloader(),
                epochs=self.config.epochs,
            )
            trainer.load_best_checkpoint()

            evaluator = MultiTaskEvaluator(self.device)
            evaluation = evaluator.evaluate(model, data_module.test_dataloader())
            self.plotter.plot_training_history(history, spec.name)
            self.plotter.plot_confusion_matrix(evaluation, spec.name)
            self.plotter.plot_age_predictions(evaluation, spec.name)

            sizes = data_module.split_sizes()
            metrics = dict(evaluation.metrics)
            metrics.update(
                {
                    "train_samples": sizes["train"],
                    "validation_samples": sizes["validation"],
                    "test_samples": sizes["test"],
                }
            )
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=self._count_trainable_parameters(model),
                training_seconds=training_seconds,
                checkpoint=str(checkpoint_path),
                message="",
            )
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    def _run_classical_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        """Entrena y evalúa el baseline clásico (PCA + LogisticRegression/Ridge).

        Args:
            spec: Especificación del experimento clásico a ejecutar.

        Returns:
            `ExperimentResult` con las métricas, o con estado de error si la
            ejecución falla.
        """
        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            data_module = UTKFaceDataModule(self.config, use_augmentation=False)
            data_module.setup()

            X_train, gender_train, age_train = self._extract_arrays(data_module.train_dataset)
            X_test, gender_test, age_test = self._extract_arrays(data_module.test_dataset)

            model = ClassicalBaseline(n_components=spec.pca_components)
            start_time = time.perf_counter()
            model.fit(X_train, gender_train, age_train)
            training_seconds = time.perf_counter() - start_time

            gender_pred, age_pred = model.predict(X_test)
            evaluation = MultiTaskMetrics.calculate(
                gender_targets=torch.from_numpy(gender_test).long(),
                gender_predictions=torch.from_numpy(gender_pred).long(),
                age_targets=torch.from_numpy(age_test).float(),
                age_predictions=torch.from_numpy(age_pred).float(),
            )

            sizes = data_module.split_sizes()
            metrics = dict(evaluation.metrics)
            metrics.update(
                {
                    "train_samples": sizes["train"],
                    "validation_samples": sizes["validation"],
                    "test_samples": sizes["test"],
                }
            )
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=None,
                training_seconds=training_seconds,
                checkpoint="",
                message="",
            )
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    @staticmethod
    def _extract_arrays(dataset: UTKFaceDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aplana todas las imágenes de un dataset en arreglos NumPy para el baseline clásico.

        Args:
            dataset: Dataset de UTKFace a recorrer.

        Returns:
            Tupla con las imágenes aplanadas, las etiquetas de género y las
            edades, todas como `np.ndarray`.
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        images: list[torch.Tensor] = []
        genders: list[torch.Tensor] = []
        ages: list[torch.Tensor] = []
        for batch_images, batch_gender, batch_age in loader:
            images.append(batch_images)
            genders.append(batch_gender)
            ages.append(batch_age)

        flattened = torch.cat(images).reshape(len(dataset), -1).numpy()
        gender_array = torch.cat(genders).numpy()
        age_array = torch.cat(ages).numpy()
        return flattened, gender_array, age_array

    def _build_model(self, spec: ExperimentSpec) -> tuple[nn.Module, dict[str, int | float | bool]]:
        """Construye el modelo neuronal correspondiente a `spec.model_kind`.

        Args:
            spec: Especificación del experimento, que determina la arquitectura
                y sus hiperparámetros.

        Returns:
            Tupla con el modelo instanciado y los kwargs usados para construirlo
            (estos últimos se guardan en el checkpoint para poder reconstruirlo).

        Raises:
            NotImplementedError: Si `spec.model_kind` no tiene una fábrica definida.
        """
        if spec.model_kind == "cnn":
            model_kwargs = {"dropout": spec.dropout}
            return MultiTaskCNN(**model_kwargs), model_kwargs

        if spec.model_kind == "mlp":
            model = MultiTaskMLP(image_size=self.config.image_size, dropout=spec.dropout)
            return model, {"image_size": self.config.image_size, "dropout": spec.dropout}

        if spec.model_kind == "resnet_frozen":
            model = MultiTaskResNet(trainable_blocks=0, pretrained=True)
            return model, {"trainable_blocks": 0, "pretrained": True}

        if spec.model_kind == "resnet_finetuning":
            model = MultiTaskResNet(trainable_blocks=spec.trainable_blocks, pretrained=True)
            return model, {"trainable_blocks": spec.trainable_blocks, "pretrained": True}

        # TODO(alumno): extend this factory when other strategies are implemented.
        raise NotImplementedError(f"No existe una fabrica para model_kind={spec.model_kind}.")

    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        """Cuenta los parámetros del modelo con `requires_grad=True`.

        Args:
            model: Modelo de PyTorch del cual contar los parámetros.

        Returns:
            Número total de parámetros entrenables.
        """
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _not_implemented_result(spec: ExperimentSpec) -> ExperimentResult:
        """Construye el resultado de un experimento aún no implementado.

        Args:
            spec: Especificación del experimento.

        Returns:
            `ExperimentResult` con estado `NOT_IMPLEMENTED`.
        """
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="El experimento debe ser completado por los alumnos.",
        )

    @staticmethod
    def _not_executed_result(spec: ExperimentSpec) -> ExperimentResult:
        """Construye el resultado de un experimento implementado pero no seleccionado.

        Args:
            spec: Especificación del experimento.

        Returns:
            `ExperimentResult` con estado `NOT_EXECUTED`.
        """
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_EXECUTED,
            message="Implementado, pero no fue seleccionado en esta ejecucion.",
        )
