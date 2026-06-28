"""Baseline clásico con PCA (E1): estimadores de scikit-learn sobre píxeles aplanados."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge


class ClassicalBaseline:
    """Reducción de dimensionalidad con PCA seguida de estimadores independientes de género/edad.

    El género usa LogisticRegression y la edad usa Ridge. Ambos estimadores se
    ajustan sobre las mismas características proyectadas por PCA, de modo que
    el único componente compartido es el paso de reducción de dimensionalidad.
    """

    def __init__(self, n_components: int = 100) -> None:
        """Inicializa el PCA y los estimadores de género y edad.

        Args:
            n_components: Número de componentes principales que conserva el PCA.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.gender_model = LogisticRegression(max_iter=1000)
        self.age_model = Ridge()

    def fit(self, X: np.ndarray, y_gender: np.ndarray, y_age: np.ndarray) -> None:
        """Ajusta el PCA y entrena los estimadores de género y edad.

        Args:
            X: Matriz de características de entrada (imágenes aplanadas).
            y_gender: Etiquetas verdaderas de género.
            y_age: Edades verdaderas.
        """
        X_reduced = self.pca.fit_transform(X)
        self.gender_model.fit(X_reduced, y_gender)
        self.age_model.fit(X_reduced, y_age)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Proyecta las características con el PCA ajustado y predice género y edad.

        Args:
            X: Matriz de características de entrada (imágenes aplanadas).

        Returns:
            Tupla con las predicciones de género y de edad.
        """
        X_reduced = self.pca.transform(X)
        gender_pred = self.gender_model.predict(X_reduced)
        age_pred = self.age_model.predict(X_reduced)
        return gender_pred, age_pred
