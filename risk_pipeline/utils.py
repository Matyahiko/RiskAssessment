import ast
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path('/app/iiai/artifacts')
ART_DIR = Path('/app/iiai/artifacts2')
SEED = 42

TARGET_CODES = [33, 25, 36, 16, 15, 7]
code2name = {
    33: 'Chemicals',
    25: 'Machinery',
    36: 'Transportation',
    16: 'Textile',
    15: 'Food',
    7: 'Mining'
}
TARGET_INDUSTRIES = [code2name[c] for c in TARGET_CODES]

FIN_COLS = [
    'fin_1', 'fin_2', 'fin_3', 'fin_4',
    'fin_5', 'fin_6', 'fin_7', 'fin_8'
]

N_TRIALS = 50

ART_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=ART_DIR / 'pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def parse_embedding(text: str) -> Optional[List[float]]:
    """Parse embedding string to list of floats."""
    try:
        vec = ast.literal_eval(text)
        if isinstance(vec, list) and len(vec) == 1536:
            return vec
    except Exception:
        logging.exception('Failed to parse embedding')
    return None


def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename)
    df['embedding'] = df['embedding'].apply(parse_embedding)
    df = df.dropna(subset=['embedding'])
    return df


def ensure_industry_dir(industry: str) -> Path:
    path = ART_DIR / industry
    path.mkdir(parents=True, exist_ok=True)
    (path / 'optuna').mkdir(exist_ok=True)
    return path
