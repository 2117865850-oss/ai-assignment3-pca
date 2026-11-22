"""Signflip component selection for two-dimensional PCA, after Li & Kuang."""
import numpy as np


def fit_2dpca(images):
    centered = images - images.mean(axis=0)
    rows = centered.reshape(-1, images.shape[-1])
    covariance = rows.T @ rows / len(images)
    values, vectors = np.linalg.eigh(covariance)
    return covariance, values[::-1], vectors[:, ::-1]


def null_spectra(covariance, trials, seed):
    rng = np.random.default_rng(seed)
    spectra = []
    for _ in range(trials):
        signs = 2 * rng.integers(0, 2, size=covariance.shape) - 1
        # Independent signs destroy symmetry. The authors' notebook sorts
        # general eigenvalues and stores them in a real-valued array.
        eigenvalues = np.linalg.eigvals(signs * covariance)
        spectra.append(np.sort(eigenvalues.real)[::-1])
    return np.stack(spectra)


def leading_above(values, thresholds):
    failures = np.flatnonzero(values <= thresholds)
    return int(failures[0]) if len(failures) else len(values)


def select_ranks(values, spectra):
    threshold = np.quantile(spectra, 0.95, axis=0, method='linear')
    fractions = np.cumsum(values) / np.sum(values)
    return {
        'SPA upper-edge': leading_above(values, threshold[0]),
        'SPA pairwise': leading_above(values, threshold),
        '90% variance': int(np.searchsorted(fractions, 0.90)) + 1,
        '95% variance': int(np.searchsorted(fractions, 0.95)) + 1,
    }, threshold


def nearest_by_rank(train, test, ranks, batch_size=125):
    """1-NN under the sum of Euclidean column distances, caching prefixes."""
    ranks = sorted(set(ranks))
    if min(ranks) < 0 or max(ranks) > train.shape[-1]:
        raise ValueError('Rank outside feature width')
    neighbors = {rank: np.zeros(len(test), dtype=int) for rank in ranks}
    # At rank zero every distance is zero, so the first training image wins.
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        distances = np.zeros((stop - start, len(train)))
        for j in range(max(ranks)):
            a = np.ascontiguousarray(test[start:stop, :, j]); b = np.ascontiguousarray(train[:, :, j])
            squared = (a * a).sum(1)[:, None] + (b * b).sum(1) - 2 * (a @ b.T)
            distances += np.sqrt(np.maximum(0, squared))
            if j + 1 in neighbors:
                neighbors[j + 1][start:stop] = distances.argmin(axis=1)
    return neighbors
