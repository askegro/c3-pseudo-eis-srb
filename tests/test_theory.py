import sys
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prbsid import sequences as sq, acquisition as ac, estimator as es, fit as ft
from prbsid.truth_ecm import EcmCell, MismatchCell

def test_sequences_are_maximum_length():
    for n in sq.PRIMITIVE_TAPS:
        b = sq.mls(n)
        N = b.size
        assert sq.is_maximum_length(b), n
        assert b.sum() == 2 ** (n - 1)
        B = np.fft.fft(b)
        assert np.isclose(abs(B[0]), (N + 1) / 2) and np.allclose(np.abs(B[1:]) ** 2, (N + 1) / 4)

def test_lemma1_line_spectrum():
    b = sq.mls(10)
    N = b.size
    OS = 256
    c = np.fft.fft(np.repeat(b, OS)) / (N * OS)
    l = np.arange(1, sq.n_lines(N) + 1)
    assert np.max(np.abs(np.abs(c[l]) / sq.line_magnitude(l, N) - 1)) < 0.0001
    assert np.isclose(c[0].real, (N + 1) / (2 * N))
    frac = 2 * np.sum(sq.line_magnitude(l, N) ** 2) / (b.mean() * (1 - b.mean()))
    assert abs(frac - 0.722) < 0.002

def test_cascade_rule():
    N = 1023
    assert sq.cascade_covers([2.0, 154.0, 10000.0], N, 0.005, 4400.0)
    assert not sq.cascade_covers([2.0, 10000.0], N, 0.005, 4400.0)
    assert sq.min_sequences(N, 0.005, 5000.0) == 3
    assert abs(np.log10(sq.HALF_POWER * N) - 2.656) < 0.001

def _discrete_system(i):
    y = 0.5 * i
    for R, t in ((1.0, 3.0), (0.7, 40.0), (1.3, 600.0)):
        a = np.exp(-1 / t)
        y = y + lfilter([0, R * (1 - a)], [1, -a], i)
    return y

def _discrete_response(w):
    z = np.exp(1j * w)
    g = 0.5 + 0 * z
    for R, t in ((1.0, 3.0), (0.7, 40.0), (1.3, 600.0)):
        a = np.exp(-1 / t)
        g = g + R * (1 - a) / (z - a)
    return g

def test_lemma2_no_leakage_at_excited_lines():
    b = sq.mls(10)
    N = b.size
    kappa = 4
    Lp = kappa * N
    i = np.tile(np.repeat(b.astype(float), kappa), 4 + 8)
    y = _discrete_system(i)[8 * Lp:]
    i = i[8 * Lp:]
    lines = np.arange(1, sq.n_lines(N) + 1)
    e = es.estimate(i, y, Lp, lines)
    G = _discrete_response(2 * np.pi * 2 * lines / (2 * Lp))
    assert np.max(np.abs(e['Z'] - G) / np.abs(G)) < 1e-09
    assert np.allclose(e['I'], e['I'][0][None, :])

def test_hann_equals_odd_bin_transient_correction():
    rng = np.random.default_rng(3)
    b = sq.mls(8)
    Lp = 4 * b.size
    x = np.tile(np.repeat(b.astype(float), 4), 3) + rng.standard_normal(3 * Lp) + np.exp(-np.arange(3 * Lp) / 300.0)
    lines = np.arange(1, 100)
    h = es.segment_transforms(x, Lp, lines, 'hann')
    r = es.segment_transforms(x, Lp, lines, 'rect', transient='odd')
    assert np.allclose(h, 0.5 * r)

def test_proposition2_variance():
    rng = np.random.default_rng(1)
    b = sq.mls(9)
    N = b.size
    kappa = 4
    Lp = kappa * N
    lines = np.arange(2, sq.n_lines(N) + 1)
    for P in (4, 8):
        K = P - 1
        i = np.tile(np.repeat(b.astype(float), kappa), P + 8)
        y = _discrete_system(i)[8 * Lp:]
        i = i[8 * Lp:]
        runs = 1500
        Z = np.empty((runs, lines.size), complex)
        S2 = np.empty((runs, lines.size))
        for m in range(runs):
            e = es.estimate(i, y + 0.05 * rng.standard_normal(y.size), Lp, lines)
            Z[m] = e['Z']
            S2[m] = e['sigma'] ** 2
        emp = 0.5 * (Z.real.var(0, ddof=1) + Z.imag.var(0, ddof=1))
        ratio = float(np.mean(S2.mean(0) / emp))
        assert abs(ratio - 1) < 0.04, (K, ratio)
        aK, ne = es.n_eff(K)
        assert np.isclose(aK, 1 + (K - 1) / (3 * K)) and np.isclose(ne, (K - aK) / aK)
    assert np.isclose(es.n_eff(3)[1], 1.4545, atol=0.0001) and np.isclose(es.n_eff(7)[1], 4.4444, atol=0.0001)

def test_remark1_convex_fit():
    rng = np.random.default_rng(2)
    cell = EcmCell()
    f = np.logspace(-2.3, 3.6, 60)
    w = 2 * np.pi * f
    Z = cell.eis_eta(f) + 2e-06 * (rng.standard_normal(60) + 1j * rng.standard_normal(60))
    a = ft.nnls_fit(w, Z)
    assert a['kkt'] < 1e-10 and np.linalg.matrix_rank(np.vstack([ft.phi(w).real, ft.phi(w).imag])) == 7
    truth = np.concatenate([[cell.r0], cell.r])
    assert np.max(np.abs(a['theta'] - truth)) < 2e-05
    for seed in range(5):
        nl = ft.nlls_fixed_tau(w, Z, rng=np.random.default_rng(seed))
        assert nl['cost'] >= a['cost'] * (1 - 1e-09)
        assert np.max(np.abs(nl['theta'] - a['theta'])) < 2e-05
    sig = np.full(60, 2e-06)
    rt = [ft.nnls_fit(w, cell.eis_eta(f) + 2e-06 * (rng.standard_normal(60) + 1j * rng.standard_normal(60)), sig)['r_tot'] for _ in range(600)]
    pred = ft.nnls_fit(w, Z, sig)['r_tot_sd']
    assert abs(np.std(rt, ddof=1) / pred - 1) < 0.15

def test_proposition1_power_under_a_time_varying_load():
    rng = np.random.default_rng(4)
    b = sq.mls(7)
    N = b.size
    OS = 8
    P = 400
    s = sq.switching_signal(b, P, OS)
    load = 20.0 + lfilter([0.02], [1, -0.98], rng.standard_normal(s.size + 2000))[2000:] * 40.0
    x = s * load
    n = x.size
    X = np.fft.rfft(x) / n
    p = 2 * np.abs(X) ** 2
    l = np.arange(3, 40)
    band_power = np.array([p[k * P - P // 2:k * P + P // 2].sum() for k in l])
    expected = 2 * sq.line_magnitude(l, N) ** 2 * np.mean(load ** 2)
    assert abs(np.mean(band_power[10:] / expected[10:]) - 1) < 0.1

def test_proposition3_pack_bound():
    I0 = 20.0
    b = sq.mls(10)
    for dsoc, rs in ((0.0, 1.0), (0.02, 1.0), (0.0, 1.1), (0.02, 1.1)):
        p, q = (EcmCell(), EcmCell(r_scale=rs))
        for fc in (2.0, 154.0, 10000.0):
            s = sq.switching_signal(b, 3, 8)
            dt = 1 / (fc * 8)
            rp = p.run(I0 * s, dt, 0.5)
            rq = q.run(I0 * (1 - s), dt, 0.5 + dsoc)
            eta = s * (rp['u_ocv'] - rp['v']) + (1 - s) * (rq['u_ocv'] - rq['v'])
            lo = I0 * min(p.r0, q.r0)
            hi = I0 * max(p.r0 + p.r.sum(), q.r0 + q.r.sum())
            assert eta.min() >= lo - 1e-12 and eta.max() <= hi + 1e-12

def test_acquisition_common_filter_cancels():
    cell = EcmCell()
    b = sq.mls(10)
    N = b.size
    I0 = 20.0
    for fc, os_sim in ((154.0, 40), (10000.0, 20)):
        q = os_sim // 4
        s = sq.switching_signal(b, 4 + 3, os_sim)
        dt = 1 / (fc * os_sim)
        r = cell.run(I0 * s, dt, 0.55, cell.steady_state(I0 * b.mean()))
        i, eta = ac.decimate_channels(q, I0 * s, r['u_ocv'] - r['v'])
        i, eta = (i[3 * N * 4:], eta[3 * N * 4:])
        lines = np.arange(5, sq.n_lines(N) + 1)
        f = lines * fc / N
        e = es.estimate(i, eta, 4 * N, lines)
        assert np.max(np.abs(e['Z'] - cell.eis_eta(f)) / np.abs(cell.eis_eta(f))) < 0.002

def test_time_domain_models_agree():
    cell = EcmCell()
    dt = 0.01
    i = np.concatenate([np.zeros(50), np.full(1000, 60.0), np.zeros(1000)])
    r = cell.run(i, dt, 0.5)
    theta = np.concatenate([[cell.r0], cell.r])
    assert np.max(np.abs(ft.simulate_eec(theta, i, dt, cell.tau, mid=True) - (r['u_ocv'] - r['v']))) < 1e-12
    pf = ft.pulse_fit(i, r['u_ocv'] - r['v'], dt, cell.tau, mid=True)
    assert np.max(np.abs(pf['theta'] - theta)) < 1e-07

def test_mismatch_cell_runs():
    cell = MismatchCell()
    f = np.logspace(-2, 3, 20)
    a = ft.nnls_fit(2 * np.pi * f, cell.eis_eta(f))
    assert a['r_tot'] > 0 and a['kkt'] < 1e-08
if __name__ == '__main__':
    import time
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS  {name}  ({time.time() - t0:.1f} s)')
        except AssertionError as exc:
            failed += 1
            print(f'FAIL  {name}: {exc}')
    print(f'{len(tests) - failed} of {len(tests)} tests passed')
    sys.exit(1 if failed else 0)
