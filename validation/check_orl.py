"""Verify a published SPA rank choice on the original ORL face images."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_gallery(folder):
    training, testing = [], []
    for subject in range(1, 41):
        for index in range(1, 11):
            path = folder / f's{subject}' / f'{index}.pgm'
            with Image.open(path) as image:
                if image.size != (92, 112):
                    raise ValueError(f'Unexpected image dimensions in {path.name}')
                pixels = np.asarray(image, dtype=np.float64)
            group = training if index <= 5 else testing
            group.append(pixels)
    return np.stack(training), np.stack(testing)


def compare(folder):
    train, test = read_gallery(folder)
    residual = train - train.mean(axis=0)
    covariance = (residual.transpose(0, 2, 1) @ residual).mean(axis=0)
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    rng = np.random.default_rng(2026)
    null = []
    for _ in range(100):
        mask = rng.choice([-1, 1], size=covariance.shape)
        null.append(np.sort(np.linalg.eigvals(covariance * mask).real)[::-1])
    thresholds = np.quantile(null, 0.95, axis=0, method='linear')
    truth = np.repeat(np.arange(1, 41), 5)
    result = {
        'ORL_images': len(train) + len(test), 'G_shape': list(covariance.shape),
        'train_samples': len(train), 'test_samples': len(test), 'sign_seed': 2026,
        'null_trials': len(null), 'train_first_per_subject': 5, 'test_remaining_per_subject': 5,
    }
    for rule, threshold in [('upper', thresholds[0]), ('pairwise', thresholds)]:
        prefix = np.logical_and.accumulate(values > threshold)
        rank = int(prefix.sum())
        gallery, queries = train @ vectors[:, :rank], test @ vectors[:, :rank]
        predictions = []
        for query in queries:
            distances = np.linalg.norm(gallery - query, axis=1).sum(axis=1)
            predictions.append(truth[distances.argmin()])
        hits = int(np.count_nonzero(predictions == truth))
        result[rule] = {'rank': rank, 'correct': hits, 'accuracy': hits / len(test)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Directory containing ORL s1 through s40')
    report = compare(parser.parse_args().folder)
    print(json.dumps(report, indent=2))
    expected = json.loads(Path(__file__).with_name('orl_expected.json').read_text())
    if report != expected:
        raise SystemExit('Observed ranks or recognition counts differ from the supplied check.')
