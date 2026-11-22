"""Rebuild the report's scree plot from the first training run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(results):
    with np.load(results/'fit_41.npz') as data:
        observed, thresholds = data['eigenvalues'], data['thresholds']
    ranks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7.0, 2.9), layout='constrained')
    ax.axvspan(0.7, 2.5, color='#e5eef4', zorder=0)
    ax.plot(ranks, observed[:10], color='#164161', marker='o', label='Observed eigenvalue')
    ax.plot(ranks, thresholds[:10], color='#a44722', marker='D', markersize=4, label='Pairwise threshold')
    ax.axhline(thresholds[0], color='#684778', ls='--', label='Upper-edge threshold')
    ax.set(xlabel='Component (first ten shown)', ylabel='Eigenvalue', xticks=ranks, xlim=(0.7, 10.3), ylim=(0, 35))
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='y', color='#dfdfdf', linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.savefig(results/'spectrum.png', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    plot(parser.parse_args().results)
