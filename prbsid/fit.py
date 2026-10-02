from __future__ import annotations
import numpy as np
from scipy.optimize import nnls, least_squares
from scipy.signal import lfilter
DEFAULT_TAUS = 10.0 ** np.arange(-3, 3)

def phi(omega: np.ndarray, taus: np.ndarray=DEFAULT_TAUS) -> np.ndarray:
    w = np.asarray(omega, float)[:, None]
    return np.hstack([np.ones((w.shape[0], 1), complex), 1.0 / (1.0 + 1j * w * np.asarray(taus)[None, :])])

def model_impedance(theta: np.ndarray, omega: np.ndarray, taus: np.ndarray=DEFAULT_TAUS) -> np.ndarray:
    return phi(omega, taus) @ np.asarray(theta, float)

def nnls_fit(omega, Z, sigma=None, taus=DEFAULT_TAUS) -> dict:
    P = phi(omega, taus)
    A = np.vstack([P.real, P.imag])
    y = np.concatenate([Z.real, Z.imag])
    if sigma is None:
        wts = np.ones(len(omega))
    else:
        s = np.asarray(sigma, float)
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError('sigma must be finite and positive')
        wts = 1.0 / s
    w2 = np.concatenate([wts, wts])
    Aw = A * w2[:, None]
    yw = y * w2
    scale = np.linalg.norm(Aw, axis=0)
    x, _ = nnls(Aw / scale, yw, maxiter=200 * A.shape[1])
    theta = x / scale
    free = theta > 0
    cov = np.zeros((A.shape[1], A.shape[1]))
    if free.any():
        H = Aw[:, free].T @ Aw[:, free]
        cov[np.ix_(free, free)] = np.linalg.inv(H)
        if sigma is None:
            dof = max(1, 2 * len(omega) - int(free.sum()))
            cov *= float(np.sum((Aw @ theta - yw) ** 2)) / dof
    one = np.ones(A.shape[1])
    grad = Aw.T @ (Aw @ theta - yw)
    kkt = float(max(np.max(np.abs(grad[free])) if free.any() else 0.0, np.max(np.maximum(-grad[~free], 0.0)) if (~free).any() else 0.0) / max(np.linalg.norm(Aw.T @ yw), 1e-300))
    return dict(theta=theta, cov=cov, r_tot=float(theta.sum()), r_tot_sd=float(np.sqrt(one @ cov @ one)), n_active=int((~free).sum()), kkt=kkt, cond=float(np.linalg.cond(Aw / scale)), cost=float(np.sum((Aw @ theta - yw) ** 2)))

def nlls_fixed_tau(omega, Z, taus=DEFAULT_TAUS, theta0=None, rng=None) -> dict:
    P = phi(omega, taus)
    if theta0 is None:
        rng = np.random.default_rng(0) if rng is None else rng
        theta0 = np.abs(Z).mean() * rng.uniform(0.01, 1.0, P.shape[1])
    f = lambda th: np.concatenate([(P @ th - Z).real, (P @ th - Z).imag])
    sol = least_squares(f, theta0, bounds=(0, np.inf), x_scale='jac')
    return dict(theta=sol.x, r_tot=float(sol.x.sum()), nfev=int(sol.nfev), cost=float(2 * sol.cost), success=bool(sol.success))

def nlls_free_tau(omega, Z, n_rc: int=6, rng=None) -> dict:
    rng = np.random.default_rng(0) if rng is None else rng
    w = np.asarray(omega, float)
    lo, hi = (np.log10(0.2 / w.max()), np.log10(5.0 / w.min()))
    x0 = np.concatenate([np.abs(Z).mean() * rng.uniform(0.01, 1.0, n_rc + 1), np.sort(rng.uniform(lo, hi, n_rc))])

    def f(x):
        zm = model_impedance(x[:n_rc + 1], w, 10.0 ** x[n_rc + 1:])
        return np.concatenate([(zm - Z).real, (zm - Z).imag])
    lb = np.concatenate([np.zeros(n_rc + 1), np.full(n_rc, lo)])
    ub = np.concatenate([np.full(n_rc + 1, np.inf), np.full(n_rc, hi)])
    sol = least_squares(f, x0, bounds=(lb, ub), x_scale='jac', max_nfev=4000)
    return dict(theta=sol.x[:n_rc + 1], taus=10.0 ** sol.x[n_rc + 1:], r_tot=float(sol.x[:n_rc + 1].sum()), nfev=int(sol.nfev), cost=float(2 * sol.cost), success=bool(sol.success))

def branch_responses(i: np.ndarray, dt: float, taus=DEFAULT_TAUS, mid: bool=False) -> np.ndarray:
    i = np.asarray(i, float)
    cols = [i.copy()]
    for tau in np.asarray(taus, float):
        a = np.exp(-dt / tau)
        u = lfilter([0.0, 1.0 - a], [1.0, -a], i)
        if mid:
            b = np.exp(-0.5 * dt / tau)
            u = b * u + (1.0 - b) * i
        cols.append(u)
    return np.column_stack(cols)

def simulate_eec(theta, i, dt, taus=DEFAULT_TAUS, mid: bool=False) -> np.ndarray:
    return branch_responses(i, dt, taus, mid) @ np.asarray(theta, float)

def pulse_fit(i, eta, dt, taus=DEFAULT_TAUS, mid: bool=False) -> dict:
    X = branch_responses(i, dt, taus, mid)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    x, _ = nnls(X / scale, eta, maxiter=200 * X.shape[1])
    theta = x / scale
    return dict(theta=theta, r_tot=float(theta.sum()), rmse=float(np.sqrt(np.mean((X @ theta - eta) ** 2))))

def rls_arx(i, eta, dt, order: int=2, lam: float=0.9995, p0: float=1000000.0) -> dict:
    n = order
    nt = 2 * n + 1
    th = np.zeros(nt)
    Pm = np.eye(nt) * p0
    i = np.asarray(i, float)
    eta = np.asarray(eta, float)
    for k in range(n, len(i)):
        x = np.concatenate([-eta[k - n:k][::-1], i[k - n:k + 1][::-1]])
        Px = Pm @ x
        g = Px / (lam + x @ Px)
        th = th + g * (eta[k] - x @ th)
        Pm = (Pm - np.outer(g, Px)) / lam
    a = th[:n]
    b = th[n:]
    den = 1.0 + a.sum()
    return dict(a=a, b=b, r_tot=float(b.sum() / den) if abs(den) > 1e-12 else float('nan'), r0=float(b[0]), dt=dt, stable=bool(np.all(np.abs(np.roots(np.concatenate([[1.0], a]))) < 1.0)))

def simulate_arx(model: dict, i: np.ndarray) -> np.ndarray:
    return lfilter(model['b'], np.concatenate([[1.0], model['a']]), np.asarray(i, float))
