from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from . import sequences as sq, acquisition as ac, estimator as es, fit as ft
ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None) -> dict:
    return json.loads(Path(path or ROOT / 'config.json').read_text())

def make_cell(cfg, temp_c=25.0, soh=1.0, area_scale=1.0):
    if cfg['backend'] == 'ecm':
        from .truth_ecm import EcmCell
        arr = np.exp(2500.0 * (1.0 / (273.15 + temp_c) - 1.0 / 298.15))
        return EcmCell(r_scale=arr / area_scale, soh=soh, temp_c=temp_c, capacity_ah=60.0 * area_scale)
    from .truth_pybamm import PybammCell
    pb = cfg['pybamm']
    return PybammCell(model=pb['model'], parameter_set=pb.get('parameter_set', 'Chen2020'), temp_c=temp_c, soh=soh, area_scale=area_scale, capacity_ah=pb['capacity_ah'], rtol=pb['rtol'], atol=pb['atol'], sei_fresh_m=pb['sei_fresh_m'], sei_growth_m_per_soh=pb['sei_growth_m_per_soh'], cache_dir=ROOT / 'results' / 'cache')

def band_lines(cfg, band):
    N = 2 ** cfg['sequence_degree'] - 1
    lines = np.arange(band['l_min'], sq.n_lines(N) + 1)
    return (lines, lines * band['fc'] / N)

def wltc_load(cfg, rms_a: float):
    d = np.loadtxt(ROOT / cfg['drive']['profile'], delimiter=',', comments='#')
    shape = d[:, 1] / np.sqrt(np.mean(d[:, 1] ** 2))
    return rms_a * shape

def held_load(profile, profile_dt, t_start, n, dt):
    t_mid = t_start + (np.arange(n) + 0.5) * dt
    return profile[np.floor(t_mid / profile_dt + 1e-09).astype(int) % len(profile)]

def simulate_band(cfg, cell, band, soc_mid, load, complement=False, soc_offset=0.0):
    b = sq.mls(cfg['sequence_degree'])
    N = b.size
    os_sim, kappa, P = (band['os_sim'], cfg['kappa'], band['periods'])
    if os_sim % kappa:
        raise ValueError('os_sim must be a multiple of kappa')
    s = sq.switching_signal(b, P, os_sim, complement)
    dt = 1.0 / (band['fc'] * os_sim)
    i_fine = s * load
    dq = float(np.sum(i_fine)) * dt / 3600.0
    soc0 = soc_mid + soc_offset + 0.5 * dq / cell.capacity_ah
    r = cell.run(i_fine, dt, soc0)
    q = os_sim // kappa
    out = dict(complete=bool(r['complete']), soc0=float(soc0), soc_span=float(dq / cell.capacity_ah), dt=dt * q, period_samples=N * kappa, fs=band['fc'] * kappa)
    if not r['complete']:
        out['error'] = getattr(cell, 'last_error', 'incomplete')
        return (out, None)
    i, v, u = ac.decimate_channels(q, i_fine, r['v'], r['u_ocv'])
    out.update(i=i, v=v, u=u)
    return (out, dict(i=i_fine, v=r['v'], u=r['u_ocv'], s=s, dt=dt))

def simulate_profile(cell, segments, dt, soc0):
    i = np.concatenate([np.full(int(round(d / dt)), c, float) for c, d in segments])
    r = cell.run(i, dt, soc0)
    n = int(np.sum(np.isfinite(r['v'])))
    return dict(i=i[:n], v=r['v'][:n], u=r['u_ocv'][:n], dt=dt, complete=bool(r['complete']), n_requested=int(i.size))

def measured(rec, cfg, rng, cell=None, arm='known', bms_draw=None):
    nz = cfg['noise']
    i_m = ac.add_noise(rec['i'], nz['current_sd_a'], rng)
    v_m = ac.add_noise(rec['v'], nz['voltage_sd_v'], rng)
    if arm == 'known':
        return (i_m, rec['u'] - v_m)
    dz0, cap_err = bms_draw
    cap = cell.capacity_ah * (1.0 + cap_err) * 3600.0
    soc_hat = rec['soc0'] + dz0 - (np.cumsum(i_m) - 0.5 * i_m) * rec['dt'] / cap
    return (i_m, cell.ocv(soc_hat) - v_m)

def identify(cfg, records, lines_by_band, rng, cell=None, arm='known', bms_draw=None, weighted=True):
    f_all, Z_all, s_all, c_all, b_all = ([], [], [], [], [])
    for band in cfg['bands']:
        rec = records[band['name']]
        lines, f = lines_by_band[band['name']]
        i_m, eta = measured(rec, cfg, rng, cell, arm, bms_draw)
        e = es.estimate(i_m, eta, rec['period_samples'], lines, pool=cfg['fit']['pool'])
        f_all.append(f)
        Z_all.append(e['Z'])
        s_all.append(e['sigma'])
        c_all.append(e['coh'])
        b_all.append(np.full(f.size, cfg['bands'].index(band)))
    f = np.concatenate(f_all)
    Z = np.concatenate(Z_all)
    sig = np.concatenate(s_all)
    coh = np.concatenate(c_all)
    w = 2 * np.pi * f
    fit = ft.nnls_fit(w, Z, sig if weighted else None, np.asarray(cfg['fit']['taus_s']))
    fit_dense = ft.nnls_fit(w, Z, sig if weighted else None, np.asarray(cfg['fit']['taus_dense_s']))
    return dict(f=f, Z=Z, sigma=sig, coh=coh, band=np.concatenate(b_all), fit=fit, fit_dense=fit_dense)

def rel_err(Z, Zref):
    return np.abs(Z - Zref) / np.abs(Zref)

def simulate_validation(cfg, cell, soc, load_a):
    vp = cfg['validation']
    out = {}
    for name, amp in vp['pulses'].items():
        amp = load_a if isinstance(amp, str) else float(amp)
        out[f'pulse_{name}'] = simulate_profile(cell, [(0.0, 1.0), (amp, vp['pulse_s']), (0.0, vp['rest_s'])], vp['dt_s'], soc)
        out[f'pulse_{name}']['amplitude_a'] = amp
    a, b = (int(round(x / cfg['drive']['profile_dt_s'])) for x in vp['wltc_window_s'])
    seg = wltc_load(cfg, load_a)[a:b]
    soc0 = soc + 0.5 * float(seg.sum()) * cfg['drive']['profile_dt_s'] / 3600.0 / cell.capacity_ah
    r = cell.run(seg, cfg['drive']['profile_dt_s'], soc0)
    n = int(np.sum(np.isfinite(r['v'])))
    out['wltc'] = dict(i=seg[:n], v=r['v'][:n], u=r['u_ocv'][:n], dt=cfg['drive']['profile_dt_s'], complete=bool(r['complete']), n_requested=int(seg.size))
    return out

def validation_rmse(theta, vals, taus):
    out = {}
    for name, val in vals.items():
        if val['i'].size == 0:
            out[name] = None
            continue
        pred = ft.simulate_eec(theta, val['i'], val['dt'], taus, mid=True)
        out[name] = float(np.sqrt(np.mean((pred - (val['u'] - val['v'])) ** 2)))
    return out

def pulse_resistance(theta, taus, t_s):
    g = np.concatenate([[1.0], 1.0 - np.exp(-t_s / np.asarray(taus, float))])
    return (float(g @ np.asarray(theta, float)), g)

def fit_summary(fr, taus, cfg, w, zref, vals):
    r10, g = pulse_resistance(fr['theta'], taus, cfg['fit']['pulse_resistance_s'])
    d = dict(theta=fr['theta'], r_tot=fr['r_tot'], r_pulse=r10, validation_rmse_v=validation_rmse(fr['theta'], vals, taus))
    for k in ('r_tot_sd', 'n_active', 'kkt', 'cond'):
        if k in fr:
            d[k] = fr[k]
    if 'cov' in fr:
        d['r_pulse_sd'] = float(np.sqrt(g @ fr['cov'] @ g))
    if zref is not None:
        zm = ft.model_impedance(fr['theta'], w, taus)
        d['fit_rel_err_to_ref'] = float(np.mean(rel_err(zm, zref)))
    return d

def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o

def run_opoint(cfg, task, rng, log=print):
    soc, temp_c, soh = (task['soc'], task['temp_c'], task.get('soh', 1.0))
    I0 = task.get('load_current_a', cfg['load_current_a'])
    taus = np.asarray(cfg['fit']['taus_s'])
    cell = make_cell(cfg, temp_c, soh)
    res = dict(task=task, cell=getattr(cell, 'meta', dict(kind='ecm')), capacity_ah=cell.capacity_ah, bands={})
    arrays = {}
    lines_by_band, records, zref = ({}, {}, [])
    t0 = time.time()
    for band in cfg['bands']:
        lines, f = band_lines(cfg, band)
        lines_by_band[band['name']] = (lines, f)
        zref.append(cell.eis_eta(f, soc))
    zref = np.concatenate(zref)
    log(f'  reference EIS: {time.time() - t0:.1f} s')
    for band in cfg['bands']:
        t0 = time.time()
        rec, _ = simulate_band(cfg, cell, band, soc, I0)
        res['bands'][band['name']] = {k: rec[k] for k in ('complete', 'soc0', 'soc_span', 'dt', 'period_samples', 'fs')} | {'error': rec.get('error'), 'seconds': time.time() - t0}
        log(f"  band {band['name']}: {time.time() - t0:.1f} s, complete={rec['complete']}, SOC span {100 * rec['soc_span']:.2f} %")
        if not rec['complete']:
            res['status'] = 'incomplete'
            return (res, arrays)
        records[band['name']] = rec
    pt = cfg['pulse_test']
    dense = np.asarray(cfg['fit']['taus_dense_s'])
    t0 = time.time()
    pulse = simulate_profile(cell, [(0.0, 1.0), (pt['current_a'], pt['discharge_s']), (0.0, pt['rest_s']), (-pt['regen_fraction'] * pt['current_a'], pt['discharge_s']), (0.0, pt['rest_s'])], pt['dt_s'], soc)
    vals = simulate_validation(cfg, cell, soc, I0)
    res['pulse_complete'] = pulse['complete']
    res['validation'] = {k: dict(complete=v['complete'], fraction=v['i'].size / max(1, v['n_requested']), amplitude_a=v.get('amplitude_a'), rms_overpotential_v=float(np.sqrt(np.mean((v['u'] - v['v']) ** 2))) if v['i'].size else None, rms_current_a=float(np.sqrt(np.mean(v['i'] ** 2))) if v['i'].size else None) for k, v in vals.items()}
    log(f"  pulse test and validation records: {time.time() - t0:.1f} s (complete: {', '.join((k + '=' + str(v['complete']) for k, v in vals.items()))})")
    bms_draw = (float(rng.normal(0, cfg['bms']['soc_error_sd'])), cfg['bms']['capacity_error'])
    seed_primary = int(rng.integers(2 ** 31))
    ident = identify(cfg, records, lines_by_band, np.random.default_rng(seed_primary), cell)
    ident_bms = identify(cfg, records, lines_by_band, np.random.default_rng(seed_primary), cell, 'bms', bms_draw)
    ident_unw = identify(cfg, records, lines_by_band, np.random.default_rng(seed_primary), cell, weighted=False)
    clean_cfg = dict(cfg, noise=dict(voltage_sd_v=0.0, current_sd_a=0.0))
    ident0 = identify(clean_cfg, records, lines_by_band, np.random.default_rng(0), cell, weighted=False)
    f, w = (ident['f'], 2 * np.pi * ident['f'])
    S = lambda fr, tt: fit_summary(fr, tt, cfg, w, zref, vals)
    eis6, eis_d = (ft.nnls_fit(w, zref, None, taus), ft.nnls_fit(w, zref, None, dense))
    summary = {'proposed': S(ident['fit'], taus), 'proposed_dense': S(ident['fit_dense'], dense), 'proposed_bms_ocv': S(ident_bms['fit'], taus), 'proposed_bms_ocv_dense': S(ident_bms['fit_dense'], dense), 'unweighted': S(ident_unw['fit'], taus), 'unweighted_dense': S(ident_unw['fit_dense'], dense), 'noise_free': S(ident0['fit'], taus), 'noise_free_dense': S(ident0['fit_dense'], dense), 'eis_reference': S(eis6, taus), 'eis_reference_dense': S(eis_d, dense)}
    t0 = time.time()
    nl = ft.nlls_fixed_tau(w, ident['Z'], taus, rng=np.random.default_rng(seed_primary))
    nl_s = time.time() - t0
    t0 = time.time()
    ft.nnls_fit(w, ident['Z'], None, taus)
    nnls_s = time.time() - t0
    summary['nlls_fixed_tau'] = S(nl, taus) | dict(nfev=nl['nfev'], seconds=nl_s, nnls_seconds=nnls_s, max_abs_diff_to_unweighted_nnls=float(np.max(np.abs(nl['theta'] - ident_unw['fit']['theta']))), cost_ratio_to_unweighted_nnls=float(nl['cost'] / ident_unw['fit']['cost']) if ident_unw['fit']['cost'] > 0 else None)
    free = [ft.nlls_free_tau(w, ident['Z'], len(taus), np.random.default_rng(seed_primary + k)) for k in range(cfg['fit']['free_tau_starts'])]
    summary['nlls_free_tau'] = [fit_summary(r, r['taus'], cfg, w, zref, vals) | dict(taus=r['taus'], cost=r['cost'], nfev=r['nfev'], success=r['success']) for r in free]
    prng = np.random.default_rng(seed_primary + 1)
    p_i = ac.add_noise(pulse['i'], cfg['noise']['current_sd_a'], prng)
    p_eta = pulse['u'] - ac.add_noise(pulse['v'], cfg['noise']['voltage_sd_v'], prng)
    pf6, pfd = (ft.pulse_fit(p_i, p_eta, pulse['dt'], taus, mid=True), ft.pulse_fit(p_i, p_eta, pulse['dt'], dense, mid=True))
    summary['pulse_test'] = S(pf6, taus) | dict(fit_rmse_v=pf6['rmse'])
    summary['pulse_test_dense'] = S(pfd, dense) | dict(fit_rmse_v=pfd['rmse'])
    res['fits'] = summary
    e = rel_err(ident['Z'], zref)
    e0 = rel_err(ident0['Z'], zref)
    res['impedance'] = dict(n_lines=int(f.size), mean_rel_err=float(e.mean()), median_rel_err=float(np.median(e)), max_rel_err=float(e.max()), noise_free_mean_rel_err=float(e0.mean()), noise_free_median_rel_err=float(np.median(e0)), mean_coherence=float(ident['coh'].mean()), min_coherence=float(ident['coh'].min()), per_band={b['name']: dict(mean_rel_err=float(e[ident['band'] == k].mean()), noise_free_mean_rel_err=float(e0[ident['band'] == k].mean()), mean_coherence=float(ident['coh'][ident['band'] == k].mean()), K=b['periods'] - 1, n_e=es.n_eff(b['periods'] - 1)[1]) for k, b in enumerate(cfg['bands'])})
    res['bms_draw'] = bms_draw
    arrays.update(f=f, band=ident['band'], zref=zref, z=ident['Z'], sigma=ident['sigma'], coh=ident['coh'], z_noise_free=ident0['Z'], z_bms=ident_bms['Z'], theta=ident['fit']['theta'], theta_dense=ident['fit_dense']['theta'], theta_eis=eis6['theta'], theta_eis_dense=eis_d['theta'], theta_pulse=pf6['theta'], theta_pulse_dense=pfd['theta'], i_slow=records[cfg['bands'][0]['name']]['i'])
    for k, v in vals.items():
        arrays[f'val_{k}_i'] = v['i']
        arrays[f'val_{k}_eta'] = v['u'] - v['v']
    n_mc = int(task.get('monte_carlo', 0))
    if n_mc > 0:
        t0 = time.time()
        Zs = np.empty((n_mc, f.size), complex)
        S2 = np.empty((n_mc, f.size))
        S2raw = np.empty((n_mc, f.size))
        tp = cfg['fit']['pulse_resistance_s']
        g6 = pulse_resistance(np.zeros(taus.size + 1), taus, tp)[1]
        gd = pulse_resistance(np.zeros(dense.size + 1), dense, tp)[1]
        q = {k: np.empty(n_mc) for k in ('rt', 'rt_sd', 'rp', 'rp_sd', 'rpd', 'rpd_sd', 'rt_unw', 'rpd_unw')}
        raw_cfg = dict(cfg, fit=dict(cfg['fit'], pool=0))
        for m in range(n_mc):
            seed = int(rng.integers(2 ** 31))
            idm = identify(cfg, records, lines_by_band, np.random.default_rng(seed), cell)
            Zs[m] = idm['Z']
            S2[m] = idm['sigma'] ** 2
            q['rt'][m] = idm['fit']['r_tot']
            q['rt_sd'][m] = idm['fit']['r_tot_sd']
            q['rp'][m] = g6 @ idm['fit']['theta']
            q['rp_sd'][m] = np.sqrt(g6 @ idm['fit']['cov'] @ g6)
            q['rpd'][m] = gd @ idm['fit_dense']['theta']
            q['rpd_sd'][m] = np.sqrt(gd @ idm['fit_dense']['cov'] @ gd)
            idr = identify(raw_cfg, records, lines_by_band, np.random.default_rng(seed), cell, weighted=False)
            S2raw[m] = idr['sigma'] ** 2
            q['rt_unw'][m] = idr['fit']['r_tot']
            q['rpd_unw'][m] = gd @ idr['fit_dense']['theta']
        emp = 0.5 * (Zs.real.var(axis=0, ddof=1) + Zs.imag.var(axis=0, ddof=1))
        z0 = ident0['Z']
        cover = lambda s2, ref: float(np.mean(np.concatenate([np.abs(Zs.real - ref.real) <= 2 * np.sqrt(s2), np.abs(Zs.imag - ref.imag) <= 2 * np.sqrt(s2)])))
        per_band = {}
        for k, b in enumerate(cfg['bands']):
            sel = ident['band'] == k
            per_band[b['name']] = dict(variance_ratio_pooled=float(np.mean(S2[:, sel].mean(axis=0) / emp[sel])), variance_ratio_per_line=float(np.mean(S2raw[:, sel].mean(axis=0) / emp[sel])), bias_over_sd=float(np.mean(np.abs(Zs[:, sel].mean(axis=0) - z0[sel]) / np.sqrt(2 * emp[sel] / n_mc))))
        stat = lambda v, sd: dict(mean=float(v.mean()), sd_empirical=float(v.std(ddof=1)), sd_predicted_mean=float(sd.mean()), coverage_2sd=float(np.mean(np.abs(v - v.mean()) <= 2 * sd)))
        res['monte_carlo'] = dict(runs=n_mc, seconds=time.time() - t0, per_band=per_band, coverage_2sigma_pooled_vs_noise_free=cover(S2, z0), coverage_2sigma_per_line_vs_noise_free=cover(S2raw, z0), coverage_2sigma_pooled_vs_reference=cover(S2, zref), r_tot=stat(q['rt'], q['rt_sd']), r_pulse=stat(q['rp'], q['rp_sd']), r_pulse_dense=stat(q['rpd'], q['rpd_sd']), r_tot_unweighted_sd_empirical=float(q['rt_unw'].std(ddof=1)), r_pulse_dense_unweighted_sd_empirical=float(q['rpd_unw'].std(ddof=1)))
        arrays.update(mc_emp_sd=np.sqrt(emp), mc_pred_sd=np.sqrt(S2.mean(axis=0)), mc_pred_sd_per_line=np.sqrt(S2raw.mean(axis=0)))
        log(f'  Monte Carlo ({n_mc} runs): {time.time() - t0:.1f} s')
    res['status'] = 'complete'
    return (res, arrays)

def run_drive(cfg, task, rng, log=print):
    soc, temp_c = (task['soc'], task['temp_c'])
    I_rms = task.get('load_current_a', cfg['load_current_a'])
    taus = np.asarray(cfg['fit']['taus_s'])
    dense = np.asarray(cfg['fit']['taus_dense_s'])
    dr = cfg['drive']
    cell = make_cell(cfg, temp_c, 1.0)
    profile = wltc_load(cfg, I_rms)
    res = dict(task=task, capacity_ah=cell.capacity_ah, load=dict(rms_a=float(np.sqrt(np.mean(profile ** 2))), mean_a=float(profile.mean()), max_a=float(profile.max()), min_a=float(profile.min())))
    arrays = {}
    t0 = time.time()
    vals = simulate_validation(cfg, cell, soc, I_rms)
    log(f'  validation records: {time.time() - t0:.1f} s')
    if task['variant'] == 'plain':
        dt = dr['profile_dt_s']
        n = int(round(cfg['validation']['wltc_window_s'][0] / dt))
        train = profile[:n]
        soc0 = soc + 0.5 * float(train.sum()) * dt / 3600.0 / cell.capacity_ah
        t0 = time.time()
        r = cell.run(train, dt, soc0)
        log(f"  drive cycle without excitation ({n * dt:.0f} s): {time.time() - t0:.1f} s, complete={r['complete']}")
        if not r['complete']:
            res['status'] = 'incomplete'
            res['error'] = getattr(cell, 'last_error', None)
            return (res, arrays)
        i_m = ac.add_noise(train, cfg['noise']['current_sd_a'], rng)
        eta = r['u_ocv'] - ac.add_noise(r['v'], cfg['noise']['voltage_sd_v'], rng)
        out = {}
        for order in sorted({1, dr['rls']['order']}):
            m = ft.rls_arx(i_m, eta, dt, order, dr['rls']['forgetting'])
            rm = {k: float(np.sqrt(np.mean((ft.simulate_arx(m, v['i']) - (v['u'] - v['v'])) ** 2))) if v['i'].size and abs(v['dt'] - dt) < 1e-12 else None for k, v in vals.items()}
            out[f'rls_order_{order}'] = dict(a=m['a'], b=m['b'], r_tot=m['r_tot'], r0=m['r0'], stable=m['stable'], validation_rmse_v=rm)
        for name, tt in (('batch_fit', taus), ('batch_fit_dense', dense)):
            pf = ft.pulse_fit(i_m, eta, dt, tt, mid=True)
            out[name] = fit_summary(pf, tt, cfg, None, None, vals)
        res['no_excitation'] = out
        res['training_s'] = n * dt
        res['status'] = 'complete'
        arrays.update(i=train, eta=r['u_ocv'] - r['v'])
        return (res, arrays)
    lines_by_band, records, zref = ({}, {}, [])
    for band in cfg['bands']:
        lines, f = band_lines(cfg, band)
        lines_by_band[band['name']] = (lines, f)
        zref.append(cell.eis_eta(f, soc))
    zref = np.concatenate(zref)
    res['bands'] = {}
    for band in cfg['bands']:
        N = 2 ** cfg['sequence_degree'] - 1
        n = N * band['periods'] * band['os_sim']
        dt = 1.0 / (band['fc'] * band['os_sim'])
        load = held_load(profile, dr['profile_dt_s'], dr['band_start_s'][band['name']], n, dt)
        t0 = time.time()
        rec, _ = simulate_band(cfg, cell, band, soc, load)
        res['bands'][band['name']] = {k: rec[k] for k in ('complete', 'soc0', 'soc_span', 'dt', 'period_samples', 'fs')} | {'error': rec.get('error'), 'seconds': time.time() - t0, 'load_rms_a': float(np.sqrt(np.mean(load ** 2))), 'load_mean_a': float(load.mean()), 'zero_fraction': float(np.mean(np.abs(load) < 0.02 * I_rms))}
        log(f"  band {band['name']} under the drive cycle: {time.time() - t0:.1f} s, complete={rec['complete']}")
        if not rec['complete']:
            res['status'] = 'incomplete'
            return (res, arrays)
        records[band['name']] = rec
        arrays[f"i_{band['name']}"] = rec['i']
    seed = int(rng.integers(2 ** 31))
    ident = identify(cfg, records, lines_by_band, np.random.default_rng(seed), cell)
    clean_cfg = dict(cfg, noise=dict(voltage_sd_v=0.0, current_sd_a=0.0))
    ident0 = identify(clean_cfg, records, lines_by_band, np.random.default_rng(0), cell, weighted=False)
    e, e0 = (rel_err(ident['Z'], zref), rel_err(ident0['Z'], zref))
    w = 2 * np.pi * ident['f']
    res['impedance'] = dict(mean_rel_err=float(e.mean()), median_rel_err=float(np.median(e)), noise_free_mean_rel_err=float(e0.mean()), per_band={b['name']: dict(mean_rel_err=float(e[ident['band'] == k].mean()), noise_free_mean_rel_err=float(e0[ident['band'] == k].mean()), first_decade_mean_rel_err=float(e[(ident['band'] == k) & (ident['f'] <= 10 * ident['f'][ident['band'] == k].min())].mean()), mean_coherence=float(ident['coh'][ident['band'] == k].mean())) for k, b in enumerate(cfg['bands'])})
    res['fits'] = {'proposed': fit_summary(ident['fit'], taus, cfg, w, zref, vals), 'proposed_dense': fit_summary(ident['fit_dense'], dense, cfg, w, zref, vals), 'eis_reference': fit_summary(ft.nnls_fit(w, zref, None, taus), taus, cfg, w, zref, vals), 'eis_reference_dense': fit_summary(ft.nnls_fit(w, zref, None, dense), dense, cfg, w, zref, vals)}
    arrays.update(f=ident['f'], band=ident['band'], zref=zref, z=ident['Z'], sigma=ident['sigma'], coh=ident['coh'], z_noise_free=ident0['Z'], theta=ident['fit']['theta'], theta_dense=ident['fit_dense']['theta'], i_slow=records[cfg['bands'][0]['name']]['i'])
    res['status'] = 'complete'
    return (res, arrays)

def run_pack(cfg, task, rng, log=print):
    soc, temp_c = (task['soc'], task['temp_c'])
    case = cfg['pack']['cases'][task['case']]
    I0 = task.get('load_current_a', cfg['load_current_a'])
    taus = np.asarray(cfg['fit']['taus_s'])
    cells = dict(p=make_cell(cfg, temp_c, 1.0, 1.0), q=make_cell(cfg, temp_c, 1.0, case['area_scale_q']))
    offs = dict(p=0.0, q=case['dsoc'])
    res = dict(task=task, case=case, bands={})
    arrays = {}
    lines_by_band = {b['name']: band_lines(cfg, b) for b in cfg['bands']}
    records = dict(p={}, q={})
    fine = dict(p={}, q={})
    for band in cfg['bands']:
        for g in ('p', 'q'):
            t0 = time.time()
            rec, fr = simulate_band(cfg, cells[g], band, soc, I0, complement=g == 'q', soc_offset=offs[g])
            log(f"  band {band['name']} group {g}: {time.time() - t0:.1f} s, complete={rec['complete']}")
            if not rec['complete']:
                res['status'] = 'incomplete'
                res['error'] = rec.get('error')
                return (res, arrays)
            records[g][band['name']] = rec
            fine[g][band['name']] = fr
    ids = {}
    for g in ('p', 'q'):
        zref = np.concatenate([cells[g].eis_eta(lines_by_band[b['name']][1], soc + offs[g]) for b in cfg['bands']])
        idg = identify(cfg, records[g], lines_by_band, np.random.default_rng(int(rng.integers(2 ** 31))), cells[g])
        ids[g] = idg
        res[f'group_{g}'] = dict(theta=idg['fit']['theta'], r_tot=idg['fit']['r_tot'], r_tot_sd=idg['fit']['r_tot_sd'], r_pulse_dense=pulse_resistance(idg['fit_dense']['theta'], cfg['fit']['taus_dense_s'], cfg['fit']['pulse_resistance_s'])[0], mean_rel_err=float(rel_err(idg['Z'], zref).mean()), capacity_ah=cells[g].capacity_ah)
    th = {g: ids[g]['fit_dense']['theta'] for g in ('p', 'q')}
    r0 = {g: float(th[g][0]) for g in th}
    rdc = {g: float(th[g].sum()) for g in th}
    for band in cfg['bands']:
        fp, fq = (fine['p'][band['name']], fine['q'][band['name']])
        s = fp['s']
        v_ins = s * fp['v'] + (1 - s) * fq['v']
        u_ins = s * fp['u'] + (1 - s) * fq['u']
        eta_ins = u_ins - v_ins
        up, uq = (float(fp['u'].mean()), float(fq['u'].mean()))
        lo, hi = (I0 * min(r0.values()), I0 * max(rdc.values()))
        res['bands'][band['name']] = dict(overpotential_min_v=float(eta_ins.min()), overpotential_max_v=float(eta_ins.max()), overpotential_pp_v=float(np.ptp(eta_ins)), overpotential_bound_v=[lo, hi], overpotential_bound_pp_v=float(hi - lo), within_bound=bool(eta_ins.min() >= lo and eta_ins.max() <= hi), excess_below_v=float(max(0.0, lo - eta_ins.min())), excess_above_v=float(max(0.0, eta_ins.max() - hi)), ocv_difference_v=float(abs(up - uq)), ocv_drift_v=float(np.ptp(u_ins)), v_ins_pp_v=float(np.ptp(v_ins)), bound_pp_v=float(abs(up - uq) + hi - lo), single_group_step_v=float(np.mean(fp['v'])), duty_p=float(s.mean()), charge_p_ah=float((s * fp['i']).sum() * fp['dt'] / 3600.0), charge_q_ah=float(((1 - s) * fq['i']).sum() * fq['dt'] / 3600.0))
        if band['name'] == 'mid':
            n = min(v_ins.size, 4 * band['os_sim'] * 200)
            arrays.update(pack_t=(np.arange(n) + 0.5) * fp['dt'], pack_v_ins=v_ins[-n:], pack_v_p=fp['v'][-n:], pack_s=s[-n:])
    res['status'] = 'complete'
    return (res, arrays)
RUNNERS = dict(opoint=run_opoint, drive=run_drive, pack=run_pack)
