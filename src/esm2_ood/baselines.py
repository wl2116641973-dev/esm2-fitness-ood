"""CPU baselines: AA-44 Ridge, Site-Mean, Frozen-Feature Ridge, and Hsu Augmented Ridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from esm2_ood.features import featurize_aa_substitution
from esm2_ood.metrics import compute_mse, compute_spearman

DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _select_best_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    standardize_X: bool = True,
) -> tuple[Ridge, StandardScaler | None, float, float, float]:
    """Fit Ridge with train-only scaling and select alpha on validation Spearman (tie-break MSE)."""
    scaler = StandardScaler() if standardize_X else None
    X_tr = scaler.fit_transform(X_train) if scaler is not None else X_train
    X_va = scaler.transform(X_val) if scaler is not None else X_val

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std == 0.0:
        y_std = 1.0
    y_tr_scaled = (y_train - y_mean) / y_std

    best_key = (-float("inf"), -float("inf"))
    best_model = None
    best_alpha = alphas[0]

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr_scaled)
        va_pred = model.predict(X_va) * y_std + y_mean
        rho = compute_spearman(y_val, va_pred)
        mse = compute_mse(y_val, va_pred)
        score_key = (rho if rho is not None else -float("inf"), -mse)
        if score_key > best_key:
            best_key = score_key
            best_model = model
            best_alpha = alpha

    assert best_model is not None
    return best_model, scaler, y_mean, y_std, best_alpha


class AA44RidgeBaseline:
    """Position-agnostic 44-dim amino acid substitution Ridge baseline."""

    def __init__(self, alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> None:
        self.alphas = alphas
        self.model: Ridge | None = None
        self.scaler: StandardScaler | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.best_alpha: float | None = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "AA44RidgeBaseline":
        X_tr = featurize_aa_substitution(train_df)
        X_va = featurize_aa_substitution(val_df)
        y_tr = train_df["DMS_score"].to_numpy(dtype=np.float64)
        y_va = val_df["DMS_score"].to_numpy(dtype=np.float64)
        self.model, self.scaler, self.y_mean, self.y_std, self.best_alpha = _select_best_ridge(
            X_tr, y_tr, X_va, y_va, alphas=self.alphas, standardize_X=True
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.scaler is not None
        X_te = self.scaler.transform(featurize_aa_substitution(test_df))
        return self.model.predict(X_te) * self.y_std + self.y_mean


class SiteMeanBaseline:
    """Site-level training mean baseline (falls back to global train mean on unseen sites)."""

    def __init__(self) -> None:
        self.site_means: dict[int, float] = {}
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> "SiteMeanBaseline":
        self.global_mean = float(train_df["DMS_score"].mean())
        self.site_means = {
            int(pos): float(grp["DMS_score"].mean())
            for pos, grp in train_df.groupby("pos")
        }
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.site_means.get(int(p), self.global_mean) for p in test_df["pos"]],
            dtype=np.float64,
        )


class FrozenDiffRidgeBaseline:
    """Ridge regression on frozen ESM-2 mutation-site difference embeddings."""

    def __init__(self, alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> None:
        self.alphas = alphas
        self.model: Ridge | None = None
        self.scaler: StandardScaler | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.best_alpha: float | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "FrozenDiffRidgeBaseline":
        self.model, self.scaler, self.y_mean, self.y_std, self.best_alpha = _select_best_ridge(
            X_train, y_train, X_val, y_val, alphas=self.alphas, standardize_X=True
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.scaler is not None
        return self.model.predict(self.scaler.transform(X_test)) * self.y_std + self.y_mean


class HsuAugmentedRidgeBaseline:
    """Hsu et al. (2022) augmented Ridge: [standardized zero_shot | seen-site one-hot].

    For any variant at a position unseen during training, all position one-hot columns are 0,
    so predictions reduce exactly to an affine function of the zero-shot score:
    y_hat = a * zero_shot + b.
    """

    def __init__(self, alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> None:
        self.alphas = alphas
        self.train_positions: list[int] = []
        self.pos_to_col: dict[int, int] = {}
        self.zs_scaler = StandardScaler()
        self.model: Ridge | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.best_alpha: float | None = None

    def _build_matrix(self, positions: np.ndarray, zs_scores: np.ndarray, fit: bool) -> np.ndarray:
        zs = np.asarray(zs_scores, dtype=np.float64).reshape(-1, 1)
        zs_scaled = self.zs_scaler.fit_transform(zs) if fit else self.zs_scaler.transform(zs)
        onehot = np.zeros((len(positions), len(self.train_positions)), dtype=np.float64)
        for row_idx, pos in enumerate(positions):
            col = self.pos_to_col.get(int(pos))
            if col is not None:
                onehot[row_idx, col] = 1.0
        return np.hstack([zs_scaled, onehot])

    def fit(
        self,
        train_df: pd.DataFrame,
        zs_train: np.ndarray,
        val_df: pd.DataFrame,
        zs_val: np.ndarray,
    ) -> "HsuAugmentedRidgeBaseline":
        self.train_positions = sorted(int(p) for p in train_df["pos"].unique())
        self.pos_to_col = {p: i for i, p in enumerate(self.train_positions)}

        X_tr = self._build_matrix(train_df["pos"].to_numpy(), zs_train, fit=True)
        X_va = self._build_matrix(val_df["pos"].to_numpy(), zs_val, fit=False)
        y_tr = train_df["DMS_score"].to_numpy(dtype=np.float64)
        y_va = val_df["DMS_score"].to_numpy(dtype=np.float64)

        self.model, _, self.y_mean, self.y_std, self.best_alpha = _select_best_ridge(
            X_tr, y_tr, X_va, y_va, alphas=self.alphas, standardize_X=False
        )
        return self

    def predict(self, test_df: pd.DataFrame, zs_test: np.ndarray) -> np.ndarray:
        assert self.model is not None
        X_te = self._build_matrix(test_df["pos"].to_numpy(), zs_test, fit=False)
        return self.model.predict(X_te) * self.y_std + self.y_mean
