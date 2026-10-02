from __future__ import annotations
import numpy as np
HANN_RHO = 1.0 / 6.0
RECT_RHO = 0.5

def n_eff(K: int, rho: float=HANN_RHO) -> tuple[float, float]:
    if K < 2:
        return (1.0, float('nan'))
    aK = 1.0 + 2.0 * rho * (K - 1) / K
    return (aK, (K - aK) / aK)

def segment_transforms(x: np.ndarray, period_samples: int, lines: np.ndarray, window: str='hann', transient: str='none') -> np.ndarray:
    Lp = int(period_samples)
    Ls = 2 * Lp
    P = len(x) // Lp
    K = P - 1
    if K < 1:
        raise ValueError('at least two periods are needed')
    if window == 'hann':
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(Ls) / Ls)
    elif window == 'rect':
        w = np.ones(Ls)
    else:
        raise ValueError(window)
    bins = 2 * np.asarray(lines, int)
    out = np.empty((K, bins.size), complex)
    for r in range(K):
        X = np.fft.rfft(w * x[r * Lp:r * Lp + Ls])
        out[r] = X[bins]
        if transient == 'odd':
            if window != 'rect':
                raise ValueError("transient='odd' needs the rectangular window")
            out[r] -= 0.5 * (X[bins - 1] + X[bins + 1])
    return out

def estimate(i: np.ndarray, y: np.ndarray, period_samples: int, lines: np.ndarray, window: str='hann', transient: str='none', pool: int=0) -> dict:
    I = segment_transforms(i, period_samples, lines, window, transient)
    Y = segment_transforms(y, period_samples, lines, window, transient)
    K = I.shape[0]
    Sii = np.mean(np.abs(I) ** 2, axis=0)
    Syy = np.mean(np.abs(Y) ** 2, axis=0)
    Siy = np.mean(np.conj(I) * Y, axis=0)
    Z = Siy / Sii
    coh = np.abs(Siy) ** 2 / (Sii * Syy)
    res = np.mean(np.abs(Y - Z[None, :] * I) ** 2, axis=0)
    rho = HANN_RHO if window == 'hann' else RECT_RHO
    aK, ne = n_eff(K, rho)
    if pool > 0 and K >= 2:
        k = np.ones(2 * pool + 1)
        res_s = np.convolve(res, k, mode='same') / np.convolve(np.ones_like(res), k, mode='same')
    else:
        res_s = res
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma = np.sqrt(res_s / (2.0 * ne * Sii)) if K >= 2 else np.full(Z.shape, np.nan)
    return dict(Z=Z, coh=coh, sigma=sigma, K=K, a_K=aK, n_e=ne, Sii=Sii, Syy=Syy, res=res, I=I, Y=Y)
