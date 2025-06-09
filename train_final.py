import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from risk_pipeline.utils import ART_DIR, FIN_COLS, TARGET_CODES, code2name


def train_industry(industry: str) -> None:
    in_dir = ART_DIR / industry
    params_path = in_dir / 'lgbm_best_params.json'
    if not params_path.exists():
        return
    with open(params_path) as f:
        params = json.load(f)
    with open(in_dir / 'best_iter.txt') as f:
        best_iter = int(f.read().strip())

    df = pd.read_csv(in_dir / 'clustered_results.csv')
    X = df[FIN_COLS + ['umap_x', 'umap_y']]
    y = df['cluster']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    num_round = int(best_iter * 1.2)
    model = LGBMClassifier(
        **params,
        learning_rate=0.05,
        bagging_fraction=0.8,
        bagging_freq=3,
        objective='multiclass',
        num_class=len(y.unique()),
        random_state=42,
    )
    for train_idx, valid_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            num_boost_round=num_round,
        )
    model.booster_.save_model(str(in_dir / 'lightgbm_cluster_classifier.txt'))


def main() -> None:
    for code in TARGET_CODES:
        industry = code2name[code]
        train_industry(industry)


if __name__ == '__main__':
    main()
