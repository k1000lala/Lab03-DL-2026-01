"""Classical PCA baseline (E1): scikit-learn estimators on flattened pixels."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge


class ClassicalBaseline:
    """PCA dimensionality reduction followed by independent gender/age estimators.

    Gender uses LogisticRegression, age uses Ridge. Both estimators are fit
    on the same PCA-projected features so the only shared component is the
    dimensionality reduction step.
    """

    def __init__(self, n_components: int = 100) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.gender_model = LogisticRegression(max_iter=1000)
        self.age_model = Ridge()

    def fit(self, X: np.ndarray, y_gender: np.ndarray, y_age: np.ndarray) -> None:
        X_reduced = self.pca.fit_transform(X)
        self.gender_model.fit(X_reduced, y_gender)
        self.age_model.fit(X_reduced, y_age)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_reduced = self.pca.transform(X)
        gender_pred = self.gender_model.predict(X_reduced)
        age_pred = self.age_model.predict(X_reduced)
        return gender_pred, age_pred
