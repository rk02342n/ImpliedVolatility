"""Evaluation: metrics in true IV units (not log space) and figures (AUDIT.md §3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402


def iv_metrics(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict[str, float]:
    """Invert the log1p transform and score in IV units."""
    iv_true = np.expm1(y_log_true)
    iv_pred = np.expm1(y_log_pred)
    return {
        "iv_mae": float(mean_absolute_error(iv_true, iv_pred)),
        "iv_rmse": float(np.sqrt(mean_squared_error(iv_true, iv_pred))),
        "iv_r2": float(r2_score(iv_true, iv_pred)),
        # R^2 in the (log) space the model was trained on, for reference.
        "log_iv_r2": float(r2_score(y_log_true, y_log_pred)),
    }


def save_iv_distributions(iv: pd.Series, log_iv: pd.Series, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(iv, bins=60, color="#4477aa")
    axes[0].set_title("Implied volatility (right-skewed)")
    axes[0].set_xlabel("IV")
    axes[1].hist(log_iv, bins=60, color="#228833")
    axes[1].set_title("log1p(IV) (skew reduced)")
    axes[1].set_xlabel("log1p(IV)")
    fig.tight_layout()
    fig.savefig(out_dir / "iv_distributions.png", dpi=120)
    plt.close(fig)


def save_actual_vs_predicted(
    iv_true: np.ndarray, iv_pred: np.ndarray, out_dir: Path, model_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(iv_true, iv_pred, s=4, alpha=0.25, color="#4477aa")
    lo, hi = float(np.min(iv_true)), float(np.max(iv_true))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual IV")
    ax.set_ylabel("Predicted IV")
    ax.set_title(f"Actual vs predicted IV — {model_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted_iv.png", dpi=120)
    plt.close(fig)


def save_residuals(iv_true: np.ndarray, iv_pred: np.ndarray, out_dir: Path) -> None:
    residuals = iv_true - iv_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(iv_pred, residuals, s=4, alpha=0.25, color="#ee6677")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Predicted IV")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals vs predicted IV")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=120)
    plt.close(fig)


def save_feature_importances(
    importances: dict[str, float], out_dir: Path
) -> None:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="#66ccee")
    ax.set_title("Leakage-free feature importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importances.png", dpi=120)
    plt.close(fig)


def save_model_comparison(metrics_by_model: dict[str, dict], out_dir: Path) -> None:
    names = list(metrics_by_model)
    r2s = [metrics_by_model[n]["iv_r2"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, r2s, color="#aa3377")
    ax.set_ylabel("R² (IV units, temporal test)")
    ax.set_title("Leakage-free model comparison")
    for i, v in enumerate(r2s):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=120)
    plt.close(fig)
