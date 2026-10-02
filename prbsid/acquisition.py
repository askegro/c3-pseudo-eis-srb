from __future__ import annotations
import numpy as np
from scipy.signal import firwin, lfilter

def aa_filter(q: int, taps_per_q: int=16) -> np.ndarray:
    if q == 1:
        return np.array([1.0])
    ntaps = 2 * taps_per_q * q + 1
    return firwin(ntaps, 0.6 / q, window=('kaiser', 8.0))

def decimate(x: np.ndarray, q: int, h: np.ndarray | None=None) -> np.ndarray:
    if q == 1:
        return np.asarray(x, float).copy()
    h = aa_filter(q) if h is None else h
    d = (len(h) - 1) // 2
    y = lfilter(h, [1.0], np.concatenate([x, np.zeros(d)]))[d:]
    return y[::q]

def decimate_channels(q: int, *channels: np.ndarray) -> list[np.ndarray]:
    h = aa_filter(q)
    return [decimate(c, q, h) for c in channels]

def add_noise(x: np.ndarray, sd: float, rng: np.random.Generator) -> np.ndarray:
    return x + sd * rng.standard_normal(x.shape) if sd > 0 else x.copy()
