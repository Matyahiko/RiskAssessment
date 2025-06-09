import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import shap
from lightgbm import Booster

from risk_pipeline.utils import ART_DIR, FIN_COLS, TARGET_CODES, code2name


def _normalize_shap_by_class(vals: np.ndarray) -> np.ndarray:
    return (vals - vals.mean(axis=1, keepdims=True)) / (vals.std(axis=1, keepdims=True) + 1e-6)


def _save_feature_bar(shap_values: shap.Explanation, features: pd.DataFrame, out: Path) -> None:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=mean_abs[order], y=features.columns[order], ax=ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _save_cluster_heatmap(normalized: np.ndarray, out: Path, labels: np.ndarray, features: list) -> None:
    df = pd.DataFrame(normalized, columns=features)
    df['cluster'] = labels
    df = df.groupby('cluster').mean()
    plt.figure(figsize=(10, max(4, len(df) * 0.5)))
    sns.heatmap(df, cmap='coolwarm', center=0)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def process_industry(industry: str) -> None:
    in_dir = ART_DIR / industry
    model_path = in_dir / 'lightgbm_cluster_classifier.txt'
    if not model_path.exists():
        return
    df = pd.read_csv(in_dir / 'clustered_results.csv')
    model = Booster(model_file=str(model_path))
    X = df[FIN_COLS + ['umap_x', 'umap_y']]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    out_bar = in_dir / 'shap_summary.png'
    out_heat = in_dir / 'shap_heatmap.png'
    _save_feature_bar(shap_values, X, out_bar)
    norm = _normalize_shap_by_class(shap_values.values)
    _save_cluster_heatmap(norm, out_heat, df['cluster'].to_numpy(), X.columns.tolist())


def main() -> None:
    for code in TARGET_CODES:
        industry = code2name[code]
        process_industry(industry)


if __name__ == '__main__':
    main()
