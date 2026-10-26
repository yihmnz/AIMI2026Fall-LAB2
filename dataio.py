"""Load the distributed NPZ files without changing their train/test split."""
import hashlib
from pathlib import Path
import numpy as np


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(directory, normalization='none'):
    directory = Path(directory)
    parts, metadata = {}, {'files': {}, 'imputation': 'per-electrode training mean'}
    for split in ('train', 'test'):
        xx, yy, subjects = [], [], []
        for subject in ('S4b', 'X11b'):
            p = directory / f'{subject}_{split}.npz'
            with np.load(p, allow_pickle=False) as z:
                x, y = z['signal'], z['label'].reshape(-1)
            if x.ndim != 3 or x.shape[1:] != (750, 2) or len(x) != len(y):
                raise ValueError(f'{p}: expected signal (N,750,2) and label (N,)')
            if not np.isin(y, (1, 2)).all() or np.isinf(x).any():
                raise ValueError(f'{p}: labels must be 1/2 and signals must not contain infinity')
            xx.append(np.asarray(x.transpose(0, 2, 1)[:, None], dtype=np.float32))
            yy.append(y.astype(np.int64)-1)
            subjects.extend([subject]*len(y))
            metadata['files'][p.name] = digest_file(p)
        parts[split] = [np.concatenate(xx), np.concatenate(yy), np.asarray(subjects)]
    x, y, ids = parts['train']
    xt, yt, idst = parts['test']
    mean = np.nanmean(x, axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError('An electrode contains no finite training values')
    x = np.where(np.isnan(x), mean, x)
    xt = np.where(np.isnan(xt), mean, xt)
    metadata['imputation_means'] = mean.reshape(-1).tolist()
    metadata['nan_policy'] = 'statistics fitted on training split only'
    if normalization == 'global':
        mu = x.mean(axis=(0, 3), keepdims=True, dtype=np.float64)
        sd = np.maximum(x.std(axis=(0, 3), keepdims=True, dtype=np.float64), 1e-8)
        x, xt = (x-mu)/sd, (xt-mu)/sd
        metadata['normalization_statistics'] = {'mean': mu.reshape(-1).tolist(), 'std': sd.reshape(-1).tolist()}
    metadata.update(train_shape=list(x.shape), test_shape=list(xt.shape),
                    train_counts=np.bincount(y, minlength=2).tolist(),
                    test_counts=np.bincount(yt, minlength=2).tolist(), normalization=normalization)
    return (np.ascontiguousarray(x, dtype=np.float32), y, ids,
            np.ascontiguousarray(xt, dtype=np.float32), yt, idst, metadata)
