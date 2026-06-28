"""The forecasting model: ridge regression with prediction intervals.

We fit a regularised linear model that maps engineered features (trend,
seasonality, day-of-week, demographics) to daily census. Ridge regression is a
deliberate choice for a starter system:

- It is **stable with limited data** — regularisation prevents wild swings when
  history is short, which matters early on when you have only a year or two.
- It is **interpretable** — every coefficient says how much one feature moves
  census, so the model's reasoning is inspectable, not a black box.
- It is **fast to retrain**, which is exactly what the "keep adding data and
  re-run" workflow needs.

As your history grows you can swap in a richer model behind the same interface
(``fit`` / ``predict``) without touching the rest of the system.
"""

from __future__ import annotations

import numpy as np


class RidgeModel:
    """Ridge regression with an unpenalised intercept and internal scaling.

    Features are standardised internally (so the regularisation strength applies
    evenly across features of different units), and the intercept is fit
    separately so it is never shrunk toward zero.
    """

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None  # includes intercept at index 0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.resid_std_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard constant columns
        Xs = (X - self.mean_) / self.std_

        # Prepend an intercept column of ones.
        Xa = np.column_stack([np.ones(n), Xs])

        # Ridge penalty matrix: penalise every coefficient except the intercept.
        reg = self.alpha * np.eye(d + 1)
        reg[0, 0] = 0.0

        # Solve the normal equations (X'X + reg) b = X'y.
        A = Xa.T @ Xa + reg
        b = Xa.T @ y
        self.coef_ = np.linalg.solve(A, b)

        # Residual standard deviation drives the prediction intervals.
        resid = y - Xa @ self.coef_
        dof = max(n - (d + 1), 1)
        self.resid_std_ = float(np.sqrt(np.sum(resid**2) / dof))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xa = np.column_stack([np.ones(X.shape[0]), Xs])
        return Xa @ self.coef_

    def coefficients(self, feature_names: list[str]) -> dict[str, float]:
        """Return a name→coefficient map (on the standardised feature scale)."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet; call fit() first.")
        names = ["intercept", *feature_names]
        return {name: float(c) for name, c in zip(names, self.coef_)}
