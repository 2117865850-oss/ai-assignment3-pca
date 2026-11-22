"""Run the paired Fashion-MNIST image recognition experiment."""
import argparse
import csv
import json
import platform
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from dataset import HASHES, load_split, sample_per_class
from spa import fit_2dpca, nearest_by_rank, null_spectra, select_ranks

TRAIN_SEEDS = (41, 42, 43)
NULL_SEEDS = (501, 502, 503)
TEST_SEED = 101
PER_CLASS = 100
TRIALS = 100


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    train_data, train_labels = load_split(data_dir, 'train')
    test_data, test_labels = load_split(data_dir, 't10k')
    test_ids = sample_per_class(test_labels, PER_CLASS, TEST_SEED)
    test = test_data[test_ids].astype(np.float64) / 255
    truth = test_labels[test_ids]
    rows, predictions = [], []
    for seed, null_seed in zip(TRAIN_SEEDS, NULL_SEEDS):
        train_ids = sample_per_class(train_labels, PER_CLASS, seed)
        train = train_data[train_ids].astype(np.float64) / 255
        covariance, values, axes = fit_2dpca(train)
        spectra = null_spectra(covariance, TRIALS, null_seed)
        ranks, threshold = select_ranks(values, spectra)
        width = train.shape[-1]
        ranks['Full 2DPCA'] = width
        # Project the original images, as in the authors' recognition code.
        neighbors = nearest_by_rank(train @ axes, test @ axes, ranks.values())
        by_method = {method: neighbors[rank] for method, rank in ranks.items()}
        ranks['Raw pixels'] = width
        by_method['Raw pixels'] = nearest_by_rank(train, test, [width])[width]
        np.savez_compressed(output / f'fit_{seed}.npz', train_ids=train_ids, test_ids=test_ids,
                            covariance=covariance, eigenvalues=values, eigenvectors=axes,
                            null_eigenvalues=spectra, thresholds=threshold)
        for method, nn in by_method.items():
            rank = ranks[method]
            labels = train_labels[train_ids[nn]]
            correct = int((labels == truth).sum())
            retained = float(values[:rank].sum() / values.sum())
            rows.append(dict(train_seed=seed, method=method, rank=rank, coefficients=rank*train.shape[1],
                             retained_variance=retained, correct=correct, total=len(truth), accuracy=correct/len(truth)))
            for test_id, label, guess, neighbor in zip(test_ids, truth, labels, train_ids[nn]):
                predictions.append(dict(train_seed=seed, method=method, test_index=test_id,
                                        true_label=label, predicted_label=guess, nearest_train_index=neighbor))
        print('Train seed', seed, [(r['method'],r['rank'],r['accuracy']) for r in rows if r['train_seed']==seed],flush=True)
    write_csv(output/'metrics.csv', rows)
    write_csv(output/'predictions.csv', predictions)
    summary = []
    for method in ranks:
        runs = [r for r in rows if r['method'] == method]
        summary.append(dict(method=method, ranks='|'.join(str(r['rank']) for r in runs),
                            mean_coefficients=mean(r['coefficients'] for r in runs),
                            mean_retained_variance=mean(r['retained_variance'] for r in runs),
                            mean_accuracy=mean(r['accuracy'] for r in runs),
                            sd_accuracy=stdev(r['accuracy'] for r in runs)))
    write_csv(output/'summary.csv', summary)
    metadata = dict(dataset='Fashion-MNIST', train_seeds=TRAIN_SEEDS, test_seed=TEST_SEED, null_seeds=NULL_SEEDS,
                    count_per_class_per_split=PER_CLASS, null_trials=TRIALS, quantile=.95, dtype='float64',
                    rng='NumPy default_rng (PCG64)', data_sha256=HASHES, python=platform.python_version(),
                    numpy=np.__version__, os=platform.system(), architecture=platform.machine())
    (output/'run_config.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    run(args.data_dir, args.out)
