"""Deterministic validation membership, without changing the training partition."""
import numpy as np


def split_validation(train_indices, heldout_indices, val_ratio=0.0, split_mode='segment'):
    train = np.asarray(train_indices, dtype=np.int64)
    heldout = np.asarray(heldout_indices, dtype=np.int64)
    ratio = float(val_ratio)
    if not np.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError('split_val_ratio must be finite and in [0, 1)')
    if ratio == 0:
        return np.empty(0, dtype=np.int64), heldout.copy()
    # Use the actual selected membership, including CSV-defined train/test counts.
    n_val = int(np.floor((len(train) + len(heldout)) * ratio + 1e-10))
    if n_val < 1 or n_val >= len(heldout):
        raise ValueError(f'split_val_ratio={ratio} produces {n_val} validation samples '
                         f'from {len(heldout)} held-out samples; require nonempty val and test')
    if split_mode == 'segment':
        positions = np.arange(n_val)
    elif split_mode == 'random':
        positions = np.random.RandomState(2027).permutation(len(heldout))[:n_val]
    elif split_mode == 'interval':
        positions = np.floor(np.arange(n_val) * len(heldout) / n_val).astype(np.int64)
    else:
        raise ValueError(f'Unsupported split_mode: {split_mode}')
    mask = np.zeros(len(heldout), dtype=bool)
    mask[positions] = True
    return heldout[mask], heldout[~mask]
