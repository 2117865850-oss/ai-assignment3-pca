"""Load the original Fashion-MNIST gzip files, with or without a local cache."""
import gzip
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

BASE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
HASHES = {
    'train-images-idx3-ubyte.gz': '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84',
    'train-labels-idx1-ubyte.gz': 'a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845',
    't10k-images-idx3-ubyte.gz': '346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073',
    't10k-labels-idx1-ubyte.gz': '67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5',
}


def load_idx(folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(BASE + name, timeout=90) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError(f'Checksum mismatch: {name}')
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as output:
            output.write(data)
        os.replace(output.name, path)
    raw = gzip.decompress(data)
    magic = struct.unpack_from('>I', raw)[0]
    if magic not in (2049, 2051):
        raise ValueError(f'Unexpected IDX header: {name}')
    dims = magic % 256
    shape = struct.unpack_from('>' + 'I' * dims, raw, 4)
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def load_split(folder, prefix):
    images = load_idx(folder, prefix + '-images-idx3-ubyte.gz')
    labels = load_idx(folder, prefix + '-labels-idx1-ubyte.gz')
    if images.shape != (len(labels), 28, 28) or set(labels) != set(range(10)):
        raise ValueError('Unexpected Fashion-MNIST dimensions or labels')
    return images, labels


def sample_per_class(labels, count, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.choice(np.flatnonzero(labels == i), count, replace=False) for i in range(10)]
    return np.sort(np.concatenate(groups))
