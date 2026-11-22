"""Audit saved measurements against the source images and an independent distance calculation."""
import argparse
import csv
from pathlib import Path
import numpy as np
from dataset import load_split


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify(data_dir, results):
    train, y_train = load_split(data_dir, 'train')
    test, y_test = load_split(data_dir, 't10k')
    predictions = read_csv(results/'predictions.csv')
    metrics = read_csv(results/'metrics.csv')
    assert len(predictions) == 18000 and len(metrics) == 18
    total_checked = 0
    for seed in (41, 42, 43):
        with np.load(results/f'fit_{seed}.npz') as fit:
            train_ids, test_ids = fit['train_ids'], fit['test_ids']
            assert len(set(train_ids)) == len(set(test_ids)) == 1000
            assert np.all(np.bincount(y_train[train_ids]) == 100)
            assert np.all(np.bincount(y_test[test_ids]) == 100)
            x = train[train_ids].astype(float)/255
            test_images = test[test_ids].astype(float)/255
            mean = x.mean(axis=0)
            covariance = sum((img-mean).T @ (img-mean) for img in x) / len(x)
            vals, vecs = fit['eigenvalues'], fit['eigenvectors']
            np.testing.assert_allclose(covariance, fit['covariance'], atol=1e-10)
            np.testing.assert_allclose(covariance @ vecs, vecs * vals, atol=1e-10)
            np.testing.assert_allclose(vecs.T @ vecs, np.eye(28), atol=1e-10)
            np.testing.assert_allclose(np.quantile(fit['null_eigenvalues'],.95,axis=0), fit['thresholds'])
            limits = fit['thresholds']
            assert (vals[:2] > limits[0]).all() and vals[2] <= limits[0]
            assert (vals[:2] > limits[:2]).all() and vals[2] <= limits[2]
            for row in [r for r in metrics if int(r['train_seed']) == seed]:
                method, rank = row['method'], int(row['rank'])
                items = [p for p in predictions if int(p['train_seed']) == seed and p['method'] == method]
                get_col = lambda name: np.array([int(p[name]) for p in items])
                nn_ids = get_col('nearest_train_index')
                np.testing.assert_array_equal(get_col('test_index'),test_ids)
                assert set(nn_ids) <= set(train_ids)
                np.testing.assert_array_equal(get_col('true_label'),y_test[test_ids])
                np.testing.assert_array_equal(get_col('predicted_label'),y_train[nn_ids])
                correct = np.sum(y_train[nn_ids] == y_test[test_ids])
                assert correct == int(row['correct']) and correct/len(items) == float(row['accuracy'])
                A = x if method == 'Raw pixels' else x @ vecs[:,:rank]
                B = test_images if method == 'Raw pixels' else test_images @ vecs[:,:rank]
                # Direct broadcasted norm, separate from the chunked Gram implementation.
                for i in range(0, len(B), 100):
                    distances = np.linalg.norm(A - B[i], axis=1).sum(axis=1)
                    assert train_ids[distances.argmin()] == nn_ids[i]
                    total_checked += 1
    print(f'Checked {len(predictions)} predictions and all 18 accuracy counts; independently recomputed {total_checked} nearest neighbours.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--results',type=Path,default=Path('results'))
    args=parser.parse_args()
    verify(args.data_dir,args.results)
