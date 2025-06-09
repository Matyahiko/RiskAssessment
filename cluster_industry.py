import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import umap
import hdbscan
from sklearn.metrics import silhouette_score

from risk_pipeline.utils import (
    DATA_DIR,
    ART_DIR,
    SEED,
    TARGET_CODES,
    code2name,
    N_TRIALS,
    ensure_industry_dir,
    load_data,
)


INPUT_CSV = 'gemini_results_2014_labeled_with_embeddings.csv'


def objective(trial: optuna.Trial, X: np.ndarray) -> float:
    n_neighbors = trial.suggest_int('n_neighbors', 5, 100)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.8)
    n_components = trial.suggest_int('n_components', 2, 10)

    min_cluster_size = trial.suggest_int('min_cluster_size', 2, 50)
    min_samples = trial.suggest_int('min_samples', 1, 20)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=SEED,
    )
    embedding = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(embedding)
    if len(set(labels)) <= 1:
        return -1.0
    score = silhouette_score(embedding, labels)
    return score


def train_best_model(df: pd.DataFrame, X: np.ndarray, params: dict, out_dir: Path) -> None:
    reducer = umap.UMAP(
        n_neighbors=params['n_neighbors'],
        min_dist=params['min_dist'],
        n_components=2,
        random_state=SEED,
    )
    embedding = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
    )
    labels = clusterer.fit_predict(embedding)
    df = df.copy()
    df['cluster'] = labels
    df['umap_x'] = embedding[:, 0]
    df['umap_y'] = embedding[:, 1]
    df.to_csv(out_dir / 'clustered_results.csv', index=False)


def process_industry(code: int, df: pd.DataFrame) -> None:
    name = code2name[code]
    logging.info('Processing %s', name)
    out_dir = ensure_industry_dir(name)
    df_ind = df[df['industry_code'] == code]
    if len(df_ind) < 10:
        logging.warning('Skip %s due to few samples', name)
        return
    X = np.vstack(df_ind['embedding'].to_numpy())

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X), n_trials=N_TRIALS)

    history = study.trials_dataframe()
    history.to_csv(out_dir / 'optuna' / 'history.csv', index=False)
    with open(out_dir / 'optuna' / 'best_score.txt', 'w') as f:
        f.write(str(study.best_value))
    with open(out_dir / 'optuna' / 'best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    train_best_model(df_ind, X, study.best_params, out_dir)


def main() -> None:
    df = load_data(INPUT_CSV)
    with Pool() as pool:
        pool.starmap(process_industry, [(c, df) for c in TARGET_CODES])


if __name__ == '__main__':
    main()
