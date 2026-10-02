from __future__ import annotations
import numpy as np
PRIMITIVE_TAPS = {3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7), 11: (11, 9), 12: (12, 11, 10, 4)}
HALF_POWER = 0.443

def mls(n: int, taps=None) -> np.ndarray:
    taps = PRIMITIVE_TAPS[n] if taps is None else tuple(taps)
    reg = [1] * n
    out = np.empty(2 ** n - 1, dtype=np.int8)
    for k in range(out.size):
        out[k] = reg[-1]
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    return out

def is_maximum_length(b: np.ndarray) -> bool:
    a = 1 - 2 * b.astype(int)
    N = a.size
    r = np.array([np.dot(a, np.roll(a, k)) for k in range(1, N)])
    return bool(np.all(r == -1))

def n_lines(N: int) -> int:
    return int(np.floor(HALF_POWER * N))

def line_frequencies(fc: float, N: int, lmax: int | None=None) -> np.ndarray:
    lmax = n_lines(N) if lmax is None else lmax
    return np.arange(1, lmax + 1) * fc / N

def line_magnitude(l, N: int) -> np.ndarray:
    return np.sqrt(N + 1) / (2 * N) * np.abs(np.sinc(np.asarray(l, float) / N))

def band(fc: float, N: int) -> tuple[float, float]:
    return (fc / N, HALF_POWER * fc)

def cascade_covers(clocks, N: int, fmin: float, fmax: float) -> bool:
    c = np.sort(np.asarray(clocks, float))
    return bool(c[0] / N <= fmin and HALF_POWER * c[-1] >= fmax and np.all(c[1:] <= HALF_POWER * N * c[:-1]))

def min_sequences(N: int, fmin: float, fmax: float) -> int:
    return int(np.ceil(np.log(fmax / fmin) / np.log(HALF_POWER * N) - 1e-12))

def switching_signal(b: np.ndarray, periods: int, samples_per_bit: int, complement: bool=False) -> np.ndarray:
    s = np.tile(np.repeat(b.astype(float), samples_per_bit), periods)
    return 1.0 - s if complement else s

def select_lines(N: int, per_decade: int | None=None) -> np.ndarray:
    lmax = n_lines(N)
    if per_decade is None:
        return np.arange(1, lmax + 1)
    g = np.unique(np.round(np.logspace(0, np.log10(lmax), int(np.ceil(per_decade * np.log10(lmax))) + 1)).astype(int))
    return g[(g >= 1) & (g <= lmax)]
