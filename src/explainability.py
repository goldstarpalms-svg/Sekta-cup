from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def native_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Return native feature importance when the estimator exposes it."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel().astype(float)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_table(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series | np.ndarray,
    scoring: str | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute sklearn permutation importance as a SHAP-free fallback."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(x.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_importance_table(model: Any, x: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Compute mean absolute SHAP values when the optional shap package is installed."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install optional dependency `shap` to use SHAP explanations.") from exc

    sample = x.sample(min(len(x), max_rows), random_state=42) if len(x) > max_rows else x
    explainer = shap.Explainer(model, sample)
    values = explainer(sample)
    raw_values = values.values
    if raw_values.ndim == 3:
        # Multi-output/classification estimators can return [rows, features, outputs].
        raw_values = raw_values[:, :, -1]
    importance = np.abs(raw_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(sample.columns), "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
