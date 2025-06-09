import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import Booster
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from risk_pipeline.utils import ART_DIR, FIN_COLS, TARGET_CODES, code2name


sns.set(style='whitegrid')


def evaluate(industry: str) -> None:
    in_dir = ART_DIR / industry
    model_path = in_dir / 'lightgbm_cluster_classifier.txt'
    if not model_path.exists():
        return
    df = pd.read_csv(in_dir / 'clustered_results.csv')
    model = Booster(model_file=str(model_path))
    X = df[FIN_COLS + ['umap_x', 'umap_y']]
    y = df['cluster']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    f1s = []
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in skf.split(X, y):
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        preds = model.predict(X_test)
        preds = preds.argmax(axis=1)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='macro'))
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    with open(in_dir / 'metrics.txt', 'w') as f:
        f.write(f'Accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}\n')
        f.write(f'MacroF1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n')

    report = classification_report(y_true_all, y_pred_all)
    with open(in_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(in_dir / 'confusion_matrix.png')
    plt.close(fig)


def main() -> None:
    for code in TARGET_CODES:
        industry = code2name[code]
        evaluate(industry)


if __name__ == '__main__':
    main()
