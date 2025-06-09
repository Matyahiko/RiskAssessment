import logging
from collections import Counter
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from risk_pipeline.utils import ART_DIR, TARGET_CODES, code2name


sns.set(style='whitegrid')


def visualize_industry(industry: str) -> Counter:
    out_dir = ART_DIR / industry
    csv_path = out_dir / 'clustered_results.csv'
    if not csv_path.exists():
        logging.warning('%s missing clustered_results.csv', industry)
        return Counter()

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='umap_x',
        y='umap_y',
        hue='cluster',
        palette='tab20',
        ax=ax,
        s=10,
    )
    ax.set_title(f'{industry} Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(out_dir / 'cluster_overview.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='risk_label', data=df, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'risk_bar.png')
    plt.close(fig)

    counter = Counter(df['cluster'])
    with open(out_dir / 'cluster_distribution.txt', 'w') as f:
        for cid, cnt in counter.items():
            ratio = cnt / len(df)
            f.write(f'{cid}\t{cnt}\t{ratio:.3f}\n')
    return counter


def main() -> None:
    for code in TARGET_CODES:
        industry = code2name[code]
        visualize_industry(industry)


if __name__ == '__main__':
    main()
