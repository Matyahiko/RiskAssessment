import json
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from risk_pipeline.utils import ART_DIR, FIN_COLS, TARGET_CODES, code2name


def tune_industry(industry: str) -> None:
    in_dir = ART_DIR / industry
    df = pd.read_csv(in_dir / 'clustered_results.csv')
    X = df[FIN_COLS + ['umap_x', 'umap_y']]
    y = df['cluster']
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
            'min_child_samples': trial.suggest_categorical(
                'min_child_samples', [10, 20, 50, 100]
            ),
            'feature_fraction': trial.suggest_categorical(
                'feature_fraction', [0.6, 0.8, 1.0]
            ),
            'learning_rate': 0.05,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'objective': 'multiclass',
            'num_class': len(y.unique()),
            'random_state': 42,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        preds = model.predict(X_valid)
        return f1_score(y_valid, preds, average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    history = study.trials_dataframe()
    history.to_csv(in_dir / 'lgbm_history.csv', index=False)
    with open(in_dir / 'lgbm_best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    with open(in_dir / 'best_iter.txt', 'w') as f:
        f.write(str(study.best_trial.number))


def __main__():
    for code in TARGET_CODES:
        industry = code2name[code]
        tune_industry(industry)


if __name__ == '__main__':
    __main__()
