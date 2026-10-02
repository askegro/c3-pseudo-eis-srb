from __future__ import annotations
import time
import numpy as np
from scipy.optimize import nnls
from . import acquisition as ac, estimator as es, fit as ft
from . import experiments as ex

def batch_fit_cov(i, eta, dt, taus, mid=True):
    X = ft.branch_responses(i, dt, taus, mid)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    x, _ = nnls(X / scale, eta, maxiter=200 * X.shape[1])
    theta = x / scale
    res = X @ theta - eta
    free = theta > 0
    n, k = (X.shape[0], int(free.sum()))
    s2 = float(res @ res) / max(1, n - k)
    cov = np.zeros((X.shape[1], X.shape[1]))
    if free.any():
        Xf = X[:, free]
        cov[np.ix_(free, free)] = s2 * np.linalg.inv(Xf.T @ Xf)
    return dict(theta=theta, cov=cov, rmse=float(np.sqrt(np.mean(res ** 2))), n_active=int((~free).sum()))

def r10_grad(taus, t_s):
    return np.concatenate([[1.0], 1.0 - np.exp(-t_s / np.asarray(taus, float))])

def stat(v, sd, ref):
    v, sd = (np.asarray(v, float), np.asarray(sd, float))
    return dict(mean=float(v.mean()), bias_pct=float(100 * (v.mean() - ref) / ref), sd_emp=float(v.std(ddof=1)), sd_emp_pct=float(100 * v.std(ddof=1) / ref), sd_pred_mean=float(sd.mean()), sd_ratio_pred_over_emp=float(sd.mean() / v.std(ddof=1)), coverage_2sd_vs_ref=float(np.mean(np.abs(v - ref) <= 2 * sd)), rmse_to_ref_pct=float(100 * np.sqrt(np.mean((v - ref) ** 2)) / ref), ref=float(ref))

def mean_val(rows):
    out = {}
    for k in rows[0]:
        a = np.array([np.nan if r[k] is None else r[k] for r in rows], float)
        out[k] = None if np.all(np.isnan(a)) else float(np.nanmean(a))
    return out

def noisy(cfg, i, u, v, rng):
    return (ac.add_noise(i, cfg['noise']['current_sd_a'], rng), u - ac.add_noise(v, cfg['noise']['voltage_sd_v'], rng))

def reference(cfg, cell, soc, taus, dense):
    zref = []
    for band in cfg['bands']:
        zref.append(cell.eis_eta(ex.band_lines(cfg, band)[1], soc))
    zref = np.concatenate(zref)
    f = np.concatenate([ex.band_lines(cfg, b)[1] for b in cfg['bands']])
    w = 2 * np.pi * f
    t10 = cfg['fit']['pulse_resistance_s']
    th6, thd = (ft.nnls_fit(w, zref, None, taus)['theta'], ft.nnls_fit(w, zref, None, dense)['theta'])
    return (zref, float(r10_grad(taus, t10) @ th6), float(r10_grad(dense, t10) @ thd))

def prbs_mc(cfg, records, lines_by_band, cell, rng, n_mc, taus, dense, ref6, refd):
    t10 = cfg['fit']['pulse_resistance_s']
    g6, gd = (r10_grad(taus, t10), r10_grad(dense, t10))
    q = {k: np.empty(n_mc) for k in ('rt', 'rt_sd', 'rp', 'rp_sd', 'rpd', 'rpd_sd')}
    for m in range(n_mc):
        idm = ex.identify(cfg, records, lines_by_band, np.random.default_rng(int(rng.integers(2 ** 31))), cell)
        q['rt'][m], q['rt_sd'][m] = (idm['fit']['r_tot'], idm['fit']['r_tot_sd'])
        q['rp'][m], q['rp_sd'][m] = (g6 @ idm['fit']['theta'], np.sqrt(g6 @ idm['fit']['cov'] @ g6))
        q['rpd'][m], q['rpd_sd'][m] = (gd @ idm['fit_dense']['theta'], np.sqrt(gd @ idm['fit_dense']['cov'] @ gd))
    return dict(runs=n_mc, r10_decade=stat(q['rp'], q['rp_sd'], ref6), r10_half_decade=stat(q['rpd'], q['rpd_sd'], refd), r_tot_decade_sd_ratio=float(q['rt_sd'].mean() / q['rt'].std(ddof=1)), r_tot_decade_sd_emp=float(q['rt'].std(ddof=1)), r_tot_decade_sd_pred=float(q['rt_sd'].mean()))

def batch_mc(cfg, rec_i, rec_u, rec_v, dt, vals, rng, n_mc, taus, dense, ref6, refd):
    t10 = cfg['fit']['pulse_resistance_s']
    out = {}
    for name, tt, ref in (('decade', taus, ref6), ('half_decade', dense, refd)):
        g = r10_grad(tt, t10)
        r10 = np.empty(n_mc)
        sd = np.empty(n_mc)
        rm = []
        act = np.empty(n_mc)
        for m in range(n_mc):
            i_m, eta = noisy(cfg, rec_i, rec_u, rec_v, rng)
            b = batch_fit_cov(i_m, eta, dt, tt)
            r10[m] = g @ b['theta']
            sd[m] = np.sqrt(g @ b['cov'] @ g)
            act[m] = b['n_active']
            rm.append(ex.validation_rmse(b['theta'], vals, tt))
        out[name] = dict(r10=stat(r10, sd, ref), validation_rmse_v_mean=mean_val(rm), mean_active_constraints=float(act.mean()))
    return out

def pulse_segments(cfg, amp, cycles):
    pt = cfg['pulse_test']
    seg = [(0.0, 1.0)]
    for _ in range(cycles):
        seg += [(amp, pt['discharge_s']), (0.0, pt['rest_s']), (-pt['regen_fraction'] * amp, pt['discharge_s']), (0.0, pt['rest_s'])]
    return seg

def run_xmap(cfg, task, rng, log=print):
    soc, temp_c = (task['soc'], task['temp_c'])
    I0 = cfg['load_current_a']
    xc = cfg['extra']
    taus = np.asarray(cfg['fit']['taus_s'])
    dense = np.asarray(cfg['fit']['taus_dense_s'])
    cell = ex.make_cell(cfg, temp_c, 1.0)
    res = dict(task=task, capacity_ah=cell.capacity_ah)
    arrays = {}
    t0 = time.time()
    zref, ref6, refd = reference(cfg, cell, soc, taus, dense)
    res['reference_r10'] = dict(decade=ref6, half_decade=refd)
    log(f'  reference EIS: {time.time() - t0:.1f} s')
    lines_by_band, records = ({}, {})
    for band in cfg['bands']:
        lines_by_band[band['name']] = ex.band_lines(cfg, band)
        t0 = time.time()
        rec, _ = ex.simulate_band(cfg, cell, band, soc, I0)
        log(f"  band {band['name']}: {time.time() - t0:.1f} s, complete={rec['complete']}")
        if not rec['complete']:
            res['status'] = 'incomplete'
            res['error'] = rec.get('error')
            return (res, arrays)
        records[band['name']] = rec
    t0 = time.time()
    vals = ex.simulate_validation(cfg, cell, soc, I0)
    res['validation_complete'] = {k: v['complete'] for k, v in vals.items()}
    log(f'  validation records: {time.time() - t0:.1f} s')
    t0 = time.time()
    res['prbs'] = prbs_mc(cfg, records, lines_by_band, cell, rng, int(xc['mc_prbs']), taus, dense, ref6, refd)
    ident = ex.identify(cfg, records, lines_by_band, np.random.default_rng(int(rng.integers(2 ** 31))), cell)
    res['prbs']['validation_rmse_v_decade'] = ex.validation_rmse(ident['fit']['theta'], vals, taus)
    res['prbs']['validation_rmse_v_half_decade'] = ex.validation_rmse(ident['fit_dense']['theta'], vals, dense)
    res['prbs']['duration_s'] = float(sum((b['periods'] * (2 ** cfg['sequence_degree'] - 1) / b['fc'] for b in cfg['bands'])))
    log(f"  PRBS Monte Carlo ({xc['mc_prbs']} runs): {time.time() - t0:.1f} s")
    res['pulse'] = {}
    for p in xc['pulse_protocols']:
        t0 = time.time()
        segs = pulse_segments(cfg, p['amp_a'], p['cycles'])
        dt = p['dt_s']
        dq = sum((c * d for c, d in segs)) / 3600.0
        soc0 = soc + (0.5 * dq / cell.capacity_ah if p['cycles'] > 1 else 0.0)
        r = ex.simulate_profile(cell, segs, dt, soc0)
        log(f"  pulse {p['name']}: {time.time() - t0:.1f} s, complete={r['complete']}")
        if not r['complete']:
            res['pulse'][p['name']] = dict(complete=False)
            continue
        out = batch_mc(cfg, r['i'], r['u'], r['v'], dt, vals, rng, int(xc['mc_pulse']), taus, dense, ref6, refd)
        out.update(complete=True, duration_s=float(sum((d for _, d in segs))), amp_a=p['amp_a'], cycles=p['cycles'], dt_s=dt, soc_span=float(abs(dq) / cell.capacity_ah))
        res['pulse'][p['name']] = out
        log(f"  pulse {p['name']} Monte Carlo: {time.time() - t0:.1f} s")
    res['status'] = 'complete'
    arrays.update(zref=zref)
    return (res, arrays)

def run_xdrive(cfg, task, rng, log=print):
    soc, temp_c = (task['soc'], task['temp_c'])
    I_rms = cfg['load_current_a']
    xc = cfg['extra']
    dr = cfg['drive']
    taus = np.asarray(cfg['fit']['taus_s'])
    dense = np.asarray(cfg['fit']['taus_dense_s'])
    cell = ex.make_cell(cfg, temp_c, 1.0)
    profile = ex.wltc_load(cfg, I_rms)
    dt = dr['profile_dt_s']
    res = dict(task=task, capacity_ah=cell.capacity_ah)
    arrays = {}
    zref, ref6, refd = reference(cfg, cell, soc, taus, dense)
    res['reference_r10'] = dict(decade=ref6, half_decade=refd)
    t0 = time.time()
    vals = ex.simulate_validation(cfg, cell, soc, I_rms)
    log(f'  validation records: {time.time() - t0:.1f} s')
    res['batch'] = {}
    a = int(round(cfg['validation']['wltc_window_s'][0] / dt))
    trainings = {'forward_1000s': profile[:a], 'reversed_2100s': np.resize(profile[::-1], int(round(xc['reversed_train_s'] / dt)))}
    for name, tr in trainings.items():
        t0 = time.time()
        soc0 = soc + 0.5 * float(tr.sum()) * dt / 3600.0 / cell.capacity_ah
        r = cell.run(tr, dt, soc0)
        log(f"  drive cycle without excitation, {name}: {time.time() - t0:.1f} s, complete={r['complete']}")
        if not r['complete']:
            res['batch'][name] = dict(complete=False)
            continue
        out = batch_mc(cfg, tr, r['u_ocv'], r['v'], dt, vals, rng, int(xc['mc_pulse']), taus, dense, ref6, refd)
        out.update(complete=True, duration_s=float(tr.size * dt), load_rms_a=float(np.sqrt(np.mean(tr ** 2))), load_mean_a=float(tr.mean()))
        res['batch'][name] = out
    lines_by_band, records = ({}, {})
    N = 2 ** cfg['sequence_degree'] - 1
    for band in cfg['bands']:
        lines_by_band[band['name']] = ex.band_lines(cfg, band)
        n = N * band['periods'] * band['os_sim']
        dtb = 1.0 / (band['fc'] * band['os_sim'])
        load = ex.held_load(profile, dt, dr['band_start_s'][band['name']], n, dtb)
        t0 = time.time()
        rec, _ = ex.simulate_band(cfg, cell, band, soc, load)
        log(f"  band {band['name']} under the drive cycle: {time.time() - t0:.1f} s, complete={rec['complete']}")
        if not rec['complete']:
            res['status'] = 'incomplete'
            res['error'] = rec.get('error')
            return (res, arrays)
        records[band['name']] = rec
    res['prbs'] = prbs_mc(cfg, records, lines_by_band, cell, rng, int(xc['mc_prbs']), taus, dense, ref6, refd)
    ident = ex.identify(cfg, records, lines_by_band, np.random.default_rng(int(rng.integers(2 ** 31))), cell)
    res['prbs']['validation_rmse_v_decade'] = ex.validation_rmse(ident['fit']['theta'], vals, taus)
    res['prbs']['validation_rmse_v_half_decade'] = ex.validation_rmse(ident['fit_dense']['theta'], vals, dense)
    res['prbs']['duration_s'] = float(sum((b['periods'] * N / b['fc'] for b in cfg['bands'])))
    res['status'] = 'complete'
    return (res, arrays)
RUNNERS = dict(xmap=run_xmap, xdrive=run_xdrive)
