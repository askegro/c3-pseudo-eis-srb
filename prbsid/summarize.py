import json, sys
from pathlib import Path
import numpy as np
from . import fit as ft
from .experiments import load_config, jsonable
from .tasks import tasks as build_tasks
ROOT = Path(__file__).resolve().parents[1]
VAL = (('pulse_load', 'pulse at the load current'), ('wltc', 'WLTC window'), ('pulse_two_c', '2C pulse'))

def load_all(tasks_dir, tasks):
    R, missing = ({}, [])
    for t in tasks:
        f = tasks_dir / f"{t['name']}.json"
        r = json.loads(f.read_text()) if f.exists() else None
        if r is None or r.get('status') != 'complete':
            missing.append((t['name'], 'no result' if r is None else r.get('status')))
        else:
            r['_npz'] = tasks_dir / f"{t['name']}.npz"
            R[t['name']] = r
    return (R, missing)

def m3(x):
    return None if x is None else 1000.0 * x

def fitrow(v):
    return dict(r_tot_mohm=m3(v['r_tot']), r_tot_sd_mohm=m3(v.get('r_tot_sd')), r_pulse_mohm=m3(v['r_pulse']), r_pulse_sd_mohm=m3(v.get('r_pulse_sd')), fit_rel_err_to_ref=v.get('fit_rel_err_to_ref'), validation_rmse_mv={k: m3(x) for k, x in v['validation_rmse_v'].items()}, theta_mohm=[m3(x) for x in v['theta']])

def main(argv):
    opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    tasks_dir = Path(opt('--tasks'))
    out = Path(opt('--out', tasks_dir.parent))
    cfg = load_config(opt('--config'))
    tasks = build_tasks(cfg, 'main')
    R, missing = load_all(tasks_dir, tasks)
    if missing and '--allow-missing' not in argv:
        print('Missing or incomplete tasks:')
        [print('  ', n, '-', s) for n, s in missing]
        return 1
    by = lambda group: [R[t['name']] for t in tasks if t['group'] == group and t['name'] in R]
    dense = np.asarray(cfg['fit']['taus_dense_s'])
    S = dict(missing=missing, backend=sorted({r['backend'] for r in R.values()}), versions=next(iter(R.values()))['versions'] if R else {}, settings=dict(load_current_a=cfg['load_current_a'], noise=cfg['noise'], bands=cfg['bands'], kappa=cfg['kappa'], monte_carlo_runs=cfg['monte_carlo_runs'], taus_s=cfg['fit']['taus_s'], taus_dense_s=cfg['fit']['taus_dense_s'], pulse_resistance_s=cfg['fit']['pulse_resistance_s']))
    nom, ag = (cfg['nominal'], cfg['aging'])
    find = lambda rs, **kw: next((r for r in rs if all((abs(r['task'].get(k, 1.0) - v) < 1e-09 for k, v in kw.items()))), None)
    rows = []
    for r in by('map'):
        f = r['fits']
        rows.append(dict(soc=r['task']['soc'], temp_c=r['task']['temp_c'], imp_err=r['impedance']['mean_rel_err'], imp_err_noise_free=r['impedance']['noise_free_mean_rel_err'], soc_span=r['bands']['slow']['soc_span'], validation=r['validation'], **{f'{q}_{k}': f[k][q] for q in ('r_tot', 'r_pulse') for k in ('proposed', 'proposed_dense', 'proposed_bms_ocv_dense', 'eis_reference', 'eis_reference_dense', 'pulse_test', 'pulse_test_dense')}, r_pulse_sd=f['proposed_dense']['r_pulse_sd'], r_tot_sd=f['proposed']['r_tot_sd'], **{f'val_{v}_{k}': f[k]['validation_rmse_v'][v] for v, _ in VAL for k in ('proposed', 'proposed_dense', 'eis_reference_dense', 'pulse_test_dense')}))
    if rows:
        a = lambda k: np.array([np.nan if x[k] is None else x[k] for x in rows], float)
        rel = lambda k, ref: np.abs(a(k) - a(ref)) / a(ref)
        med = lambda k: float(np.nanmedian(1000.0 * a(k)))
        S['maps'] = dict(points=len(rows), rows=rows, r_pulse_rel_err=dict(proposed_dense=dict(mean=float(rel('r_pulse_proposed_dense', 'r_pulse_eis_reference_dense').mean()), max=float(rel('r_pulse_proposed_dense', 'r_pulse_eis_reference_dense').max())), proposed_bms_ocv_dense=dict(mean=float(rel('r_pulse_proposed_bms_ocv_dense', 'r_pulse_eis_reference_dense').mean())), pulse_test_dense=dict(mean=float(rel('r_pulse_pulse_test_dense', 'r_pulse_eis_reference_dense').mean()), max=float(rel('r_pulse_pulse_test_dense', 'r_pulse_eis_reference_dense').max()))), r_tot_rel_err_decade_grid=dict(proposed=dict(mean=float(rel('r_tot_proposed', 'r_tot_eis_reference').mean()), max=float(rel('r_tot_proposed', 'r_tot_eis_reference').max())), pulse_test=dict(mean=float(rel('r_tot_pulse_test', 'r_tot_eis_reference').mean()))), impedance_mean_rel_err=float(a('imp_err').mean()), impedance_noise_free_mean_rel_err=float(a('imp_err_noise_free').mean()), validation_rmse_mv_median={v: {k: med(f'val_{v}_{k}') for k in ('proposed', 'proposed_dense', 'eis_reference_dense', 'pulse_test_dense')} for v, _ in VAL}, validation_incomplete=[dict(soc=x['soc'], temp_c=x['temp_c'], record=k) for x in rows for k, v in x['validation'].items() if not v['complete']])
    rn = find(by('map'), soc=nom['soc'], temp_c=nom['temp_c'])
    if rn:
        f = rn['fits']
        free = f['nlls_free_tau']
        S['nominal'] = dict(impedance=rn['impedance'], monte_carlo=rn.get('monte_carlo'), validation=rn['validation'], fits={k: fitrow(v) for k, v in f.items() if isinstance(v, dict)}, nlls_fixed_tau=dict(nfev=f['nlls_fixed_tau']['nfev'], seconds=f['nlls_fixed_tau']['seconds'], nnls_seconds=f['nlls_fixed_tau']['nnls_seconds'], max_abs_diff_to_unweighted_nnls_mohm=m3(f['nlls_fixed_tau']['max_abs_diff_to_unweighted_nnls']), cost_ratio_to_unweighted_nnls=f['nlls_fixed_tau']['cost_ratio_to_unweighted_nnls']), nlls_free_tau=dict(starts=len(free), r_tot_mohm=[m3(x['r_tot']) for x in free], r_pulse_mohm=[m3(x['r_pulse']) for x in free], validation_rmse_mv=[{k: m3(v) for k, v in x['validation_rmse_v'].items()} for x in free], cost=[x['cost'] for x in free]), nnls={k: dict(kkt=f[k]['kkt'], cond=f[k]['cond'], n_active=f[k]['n_active']) for k in ('proposed', 'proposed_dense')})
    rp = next((r for r in by('drive') if r['task']['variant'] == 'plain'), None)
    if rp:
        S['no_excitation'] = {k: dict(r_tot_mohm=m3(v['r_tot']), r_pulse_mohm=m3(v.get('r_pulse')), stable=v.get('stable'), validation_rmse_mv={a: m3(b) for a, b in v['validation_rmse_v'].items()}) for k, v in rp['no_excitation'].items()}
    fresh = find(by('map'), soc=ag['soc'], temp_c=ag['temp_c'])
    aged = sorted(by('aging'), key=lambda r: -r['task']['soh'])
    if fresh and aged:
        th_bol = np.array(fresh['fits']['proposed_dense']['theta'])
        rows = []
        for r in [fresh] + aged:
            z = np.load(r['_npz'])
            fd = r['fits']['proposed_dense']
            bol = {}
            for v, _ in VAL:
                i, eta = (z[f'val_{v}_i'], z[f'val_{v}_eta'])
                bol[v] = m3(float(np.sqrt(np.mean((ft.simulate_eec(th_bol, i, cfg['validation']['dt_s'], dense, mid=True) - eta) ** 2)))) if i.size else None
            rows.append(dict(soh=r['task'].get('soh', 1.0), capacity_ah=r['capacity_ah'], r_pulse_mohm=m3(fd['r_pulse']), r_pulse_sd_mohm=m3(fd['r_pulse_sd']), r_pulse_eis_mohm=m3(r['fits']['eis_reference_dense']['r_pulse']), r_tot_decade_mohm=m3(r['fits']['proposed']['r_tot']), r_tot_decade_sd_mohm=m3(r['fits']['proposed']['r_tot_sd']), rmse_bol_model_mv=bol, rmse_reidentified_mv={k: m3(x) for k, x in fd['validation_rmse_v'].items()}, validation=r['validation']))
        S['aging'] = rows
    amp = sorted(by('amplitude') + ([rn] if rn else []), key=lambda r: r['task'].get('load_current_a', cfg['load_current_a']))
    if amp:
        S['amplitude'] = [dict(load_a=r['task'].get('load_current_a', cfg['load_current_a']), mean_rel_err=r['impedance']['mean_rel_err'], noise_free_mean_rel_err=r['impedance']['noise_free_mean_rel_err'], soc_span=r['bands']['slow']['soc_span'], r_pulse_mohm=m3(r['fits']['proposed_dense']['r_pulse']), r_pulse_eis_mohm=m3(r['fits']['eis_reference_dense']['r_pulse']), validation_rmse_mv={k: m3(x) for k, x in r['fits']['proposed_dense']['validation_rmse_v'].items()}, eis_fit_validation_rmse_mv={k: m3(x) for k, x in r['fits']['eis_reference_dense']['validation_rmse_v'].items()}) for r in amp]
    rw = next((r for r in by('drive') if r['task']['variant'] == 'wltc'), None)
    if rw and rn:
        S['drive'] = dict(load=rw['load'], wltc=rw['impedance'], bands=rw['bands'], fits={k: fitrow(v) for k, v in rw['fits'].items()}, constant=dict(mean_rel_err=rn['impedance']['mean_rel_err'], noise_free_mean_rel_err=rn['impedance']['noise_free_mean_rel_err'], per_band=rn['impedance']['per_band'], fits={k: fitrow(rn['fits'][k]) for k in ('proposed', 'proposed_dense')}))
    if by('pack'):
        S['pack'] = {r['task']['case']: dict(bands=r['bands'], group_p=r['group_p'], group_q=r['group_q']) for r in by('pack')}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(jsonable(S), indent=1))
    fmt = lambda x, n=2: '--' if x is None or not np.isfinite(x) else f'{x:.{n}f}'
    L = []
    if 'nominal' in S:
        F = S['nominal']['fits']
        N = S['nominal']
        labels = [('proposed_dense', 'Proposed, half-decade grid'), ('proposed', 'Proposed, decade grid'), ('unweighted_dense', 'Unweighted, half-decade grid'), ('nlls_fixed_tau', 'Nonlinear routine, decade grid'), ('eis_reference_dense', 'Fit of the reference EIS, half-decade grid'), ('eis_reference', 'Fit of the reference EIS, decade grid'), ('pulse_test_dense', 'Pulse test, half-decade grid'), ('pulse_test', 'Pulse test, decade grid')]
        L += ['## Nominal point', '', f"| Fit | R_tot (mOhm) | R after {cfg['fit']['pulse_resistance_s']:g} s (mOhm) | sd | " + ' | '.join((f'RMSE, {n} (mV)' for _, n in VAL)) + ' |', '|---|---|---|---|' + '---|' * len(VAL)]
        tex = ['\\begin{tabular}{lcccc}', '\\hline', 'Method & $R_{10}$ (m$\\Omega$) & \\multicolumn{3}{c}{Voltage error (mV)}\\\\', ' & & load & WLTC & 2C\\\\', '\\hline']
        for k, lab in labels:
            v = F[k]
            L.append(f"| {lab} | {fmt(v['r_tot_mohm'], 3)} | {fmt(v['r_pulse_mohm'], 3)} | {fmt(v['r_pulse_sd_mohm'], 4)} | " + ' | '.join((fmt(v['validation_rmse_mv'][a]) for a, _ in VAL)) + ' |')
            tex.append(f"{lab} & {fmt(v['r_pulse_mohm'], 3)} & " + ' & '.join((fmt(v['validation_rmse_mv'][a]) for a, _ in VAL)) + '\\\\')
        nf = N['nlls_free_tau']
        L.append(f"| Nonlinear routine, free time constants, {nf['starts']} random starts | {min(nf['r_tot_mohm']):.3f} to {max(nf['r_tot_mohm']):.3f} | {min(nf['r_pulse_mohm']):.3f} to {max(nf['r_pulse_mohm']):.3f} | -- | " + ' | '.join((f"{min((x[a] for x in nf['validation_rmse_mv'])):.2f} to {max((x[a] for x in nf['validation_rmse_mv'])):.2f}" for a, _ in VAL)) + ' |')
        tex.append(f"Nonlinear routine, free $\\tau$ ({nf['starts']} starts) & {min(nf['r_pulse_mohm']):.3f}--{max(nf['r_pulse_mohm']):.3f} & " + ' & '.join((f"{min((x[a] for x in nf['validation_rmse_mv'])):.2f}--{max((x[a] for x in nf['validation_rmse_mv'])):.2f}" for a, _ in VAL)) + '\\\\')
        names = {'rls_order_1': 'Recursive least squares, 1 RC, no excitation', 'rls_order_2': 'Recursive least squares, 2 RC, no excitation', 'batch_fit': 'Batch fit, decade grid, no excitation', 'batch_fit_dense': 'Batch fit, half-decade grid, no excitation'}
        for k, v in S.get('no_excitation', {}).items():
            L.append(f"| {names.get(k, k)} | {fmt(v['r_tot_mohm'], 3)} | {fmt(v['r_pulse_mohm'], 3)} | -- | " + ' | '.join((fmt(v['validation_rmse_mv'][a]) for a, _ in VAL)) + ' |')
            tex.append(f"{names.get(k, k)} & {fmt(v['r_pulse_mohm'] if v['r_pulse_mohm'] is not None else v['r_tot_mohm'], 3)} & " + ' & '.join((fmt(v['validation_rmse_mv'][a]) for a, _ in VAL)) + '\\\\')
        (out / 'tables' / 'baselines.tex').write_text('\n'.join(tex + ['\\hline', '\\end{tabular}']) + '\n')
        imp, mc, vv = (N['impedance'], N['monte_carlo'], N['validation'])
        L += ['', 'Validation records: ' + '; '.join((f"{n}: rms current {fmt(vv[a]['rms_current_a'], 1)} A, rms overpotential {fmt(m3(vv[a]['rms_overpotential_v']), 1)} mV" for a, n in VAL)) + '.', f"Impedance error against the reference EIS: mean {100 * imp['mean_rel_err']:.2f} % with noise, {100 * imp['noise_free_mean_rel_err']:.2f} % noise-free (per band, noise-free: " + ', '.join((f"{b} {100 * v['noise_free_mean_rel_err']:.2f} %" for b, v in imp['per_band'].items())) + f"); mean coherence {imp['mean_coherence']:.5f}.", f"Misfit of the model to the reference EIS (mean relative): decade grid {100 * F['eis_reference']['fit_rel_err_to_ref']:.1f} %, half-decade grid {100 * F['eis_reference_dense']['fit_rel_err_to_ref']:.1f} %."]
        if mc:
            L += [f"Monte Carlo, {mc['runs']} runs. Predicted over empirical variance of the impedance estimate: " + ', '.join((f"{b} {v['variance_ratio_pooled']:.3f}" for b, v in mc['per_band'].items())) + '.', f"Two-sigma coverage: {100 * mc['coverage_2sigma_pooled_vs_noise_free']:.1f} % against the noise-free estimate (per-line variance {100 * mc['coverage_2sigma_per_line_vs_noise_free']:.1f} %), {100 * mc['coverage_2sigma_pooled_vs_reference']:.1f} % against the reference EIS.", 'Fitted resistances, empirical / predicted standard deviation (mOhm): ' + '; '.join((f"{lab} {m3(mc[k]['sd_empirical']):.4f} / {m3(mc[k]['sd_predicted_mean']):.4f} (coverage {100 * mc[k]['coverage_2sd']:.1f} %)" for k, lab in (('r_tot', 'R_tot, decade grid'), ('r_pulse', 'R_10, decade grid'), ('r_pulse_dense', 'R_10, half-decade grid')))) + '.']
        fx = N['nlls_fixed_tau']
        L += [f"Nonlinear routine with fixed time constants: largest parameter difference to the non-negative least-squares solution {fx['max_abs_diff_to_unweighted_nnls_mohm']:.2e} mOhm, cost ratio {fx['cost_ratio_to_unweighted_nnls']:.6f}, {fx['nfev']} function evaluations.", '']
    if 'maps' in S:
        m = S['maps']
        rr = m['r_pulse_rel_err']
        L += ['## Maps', '', f"{m['points']} operating points. Resistance after {cfg['fit']['pulse_resistance_s']:g} s (half-decade grid), relative difference to the fit of the reference EIS: proposed mean {100 * rr['proposed_dense']['mean']:.2f} % (max {100 * rr['proposed_dense']['max']:.2f} %), proposed with the BMS open-circuit arm {100 * rr['proposed_bms_ocv_dense']['mean']:.2f} %, pulse test mean {100 * rr['pulse_test_dense']['mean']:.2f} % (max {100 * rr['pulse_test_dense']['max']:.2f} %).", f"R_tot on the decade grid, as in the preliminary paper: proposed mean {100 * m['r_tot_rel_err_decade_grid']['proposed']['mean']:.2f} % (max {100 * m['r_tot_rel_err_decade_grid']['proposed']['max']:.2f} %), pulse test mean {100 * m['r_tot_rel_err_decade_grid']['pulse_test']['mean']:.2f} %.", f"Impedance error against the reference EIS, mean over points: {100 * m['impedance_mean_rel_err']:.2f} % (noise-free {100 * m['impedance_noise_free_mean_rel_err']:.2f} %).", 'Median voltage error over the points (mV): ' + '; '.join((f'{n}: ' + ', '.join((f'{k} {fmt(x)}' for k, x in m['validation_rmse_mv_median'][a].items())) for a, n in VAL)) + '.', 'Validation records that stopped at a voltage limit: ' + ', '.join((f"SOC {x['soc']:.1f} at {x['temp_c']:g} C ({x['record']})" for x in m['validation_incomplete'])) + '.' if m['validation_incomplete'] else 'All validation records ran to the end.', '']
    if 'aging' in S:
        L += ['## Ageing', '', f"| SOH | capacity (Ah) | R after {cfg['fit']['pulse_resistance_s']:g} s (mOhm) | 2 sd | same, EIS fit | " + ' | '.join((f'RMSE {n}: BOL model / re-identified (mV)' for _, n in VAL)) + ' |', '|---|---|---|---|---|' + '---|' * len(VAL)]
        tex = ['\\begin{tabular}{ccccc}', '\\hline', 'SOH & $\\hat R_{10}$ (m$\\Omega$) & \\multicolumn{3}{c}{Voltage error, BOL / re-identified (mV)}\\\\', ' & & load & WLTC & 2C\\\\', '\\hline']
        for r in S['aging']:
            L.append(f"| {r['soh']:.2f} | {fmt(r['capacity_ah'], 1)} | {fmt(r['r_pulse_mohm'], 3)} | {fmt(2 * r['r_pulse_sd_mohm'], 3)} | {fmt(r['r_pulse_eis_mohm'], 3)} | " + ' | '.join((f"{fmt(r['rmse_bol_model_mv'][a])} / {fmt(r['rmse_reidentified_mv'][a])}" for a, _ in VAL)) + ' |')
            tex.append(f"{r['soh']:.2f} & {fmt(r['r_pulse_mohm'], 3)} $\\pm$ {fmt(2 * r['r_pulse_sd_mohm'], 3)} & " + ' & '.join((f"{fmt(r['rmse_bol_model_mv'][a], 1)} / {fmt(r['rmse_reidentified_mv'][a], 1)}" for a, _ in VAL)) + '\\\\')
        (out / 'tables' / 'aging.tex').write_text('\n'.join(tex + ['\\hline', '\\end{tabular}']) + '\n')
        L.append('')
    if 'amplitude' in S:
        L += ['## Load amplitude', '', '| Load (A) | impedance error | noise-free | SOC span, slow band | RMSE at the load current: proposed / EIS fit (mV) |', '|---|---|---|---|---|']
        L += [f"| {r['load_a']:.0f} | {100 * r['mean_rel_err']:.2f} % | {100 * r['noise_free_mean_rel_err']:.2f} % | {100 * r['soc_span']:.1f} % | {fmt(r['validation_rmse_mv']['pulse_load'])} / {fmt(r['eis_fit_validation_rmse_mv']['pulse_load'])} |" for r in S['amplitude']] + ['']
    if 'drive' in S:
        d = S['drive']
        L += ['## Drive cycle', '', f"Load: rms {d['load']['rms_a']:.1f} A, mean {d['load']['mean_a']:.1f} A, from {d['load']['min_a']:.1f} to {d['load']['max_a']:.1f} A.", '', '| Band | load rms in the record (A) | error, WLTC | noise-free | error, constant load | noise-free | coherence, WLTC |', '|---|---|---|---|---|---|---|']
        L += [f"| {b} | {d['bands'][b]['load_rms_a']:.1f} | {100 * v['mean_rel_err']:.2f} % | {100 * v['noise_free_mean_rel_err']:.2f} % | {100 * d['constant']['per_band'][b]['mean_rel_err']:.2f} % | {100 * d['constant']['per_band'][b]['noise_free_mean_rel_err']:.2f} % | {v['mean_coherence']:.5f} |" for b, v in d['wltc']['per_band'].items()]
        L += ['', 'Voltage error of the fit identified under the WLTC load / under the constant load (half-decade grid, mV): ' + '; '.join((f"{n}: {fmt(d['fits']['proposed_dense']['validation_rmse_mv'][a])} / {fmt(d['constant']['fits']['proposed_dense']['validation_rmse_mv'][a])}" for a, n in VAL)) + '.', '']
    if 'pack' in S:
        L += ['## Pack level', '', '| Case | Band | overpotential of the inserted group, min to max (mV) | bound of Proposition 3 (mV) | inside | OCV difference (mV) | voltage p-p (mV) | step if one group is toggled alone (V) |', '|---|---|---|---|---|---|---|---|']
        for c, v in S['pack'].items():
            L += [f"| {c} | {b} | {m3(x['overpotential_min_v']):.1f} to {m3(x['overpotential_max_v']):.1f} | {m3(x['overpotential_bound_v'][0]):.1f} to {m3(x['overpotential_bound_v'][1]):.1f} | {('yes' if x['within_bound'] else 'no (' + fmt(m3(x['excess_below_v'])) + ' below, ' + fmt(m3(x['excess_above_v'])) + ' above)')} | {m3(x['ocv_difference_v']):.1f} | {m3(x['v_ins_pp_v']):.1f} | {x['single_group_step_v']:.3f} |" for b, x in v['bands'].items()]
        L += ['', 'Both groups identified in the same test, mean impedance error: ' + '; '.join((f"{c}: p {100 * v['group_p']['mean_rel_err']:.2f} %, q {100 * v['group_q']['mean_rel_err']:.2f} %" for c, v in S['pack'].items())) + '.', '']
    if missing:
        L = ['**Missing or incomplete tasks:** ' + ', '.join((n for n, _ in missing)), ''] + L
    if 'nominal' in S and 'no_excitation' in S:
        N, X = (S['nominal'], S['no_excitation'])
        f = N['fits']
        free = N['nlls_free_tau']
        vm = free['validation_rmse_mv']
        rg = lambda xs, fmt: f'{fmt % min(xs)}--{fmt % max(xs)}'

        def row(name, r10, d):
            m = d['validation_rmse_mv']
            return f"{name} & {r10} & {m['pulse_load']:.2f} & {m['wltc']:.2f} & {m['pulse_two_c']:.1f}\\\\\n"
        fr = lambda name, d: row(name, f"{d['r_pulse_mohm']:.3f}", d)
        t = '\\begin{tabular}{@{}lcccc@{}}\n\\hline\nMethod & $R_{10}$ & \\multicolumn{3}{c}{Voltage error (mV)}\\\\\n & (m$\\Omega$) & 20\\,A & WLTC & 2C\\\\\n\\hline\n'
        t += fr('Pseudo-EIS, $M{=}14$', f['proposed_dense']) + fr('Pseudo-EIS, $M{=}6$', f['proposed'])
        t += '\\hline\n'
        t += fr('Unweighted NNLS, $M{=}14$', f['unweighted_dense']) + fr('Unweighted NNLS, $M{=}6$', f['unweighted'])
        t += fr('Nonlinear routine, $M{=}6$', f['nlls_fixed_tau'])
        t += f"Nonlinear, free $\\tau$ & {rg(free['r_pulse_mohm'], '%.3f')} & {rg([x['pulse_load'] for x in vm], '%.2f')} & {rg([x['wltc'] for x in vm], '%.2f')} & {rg([x['pulse_two_c'] for x in vm], '%.1f')}\\\\\n"
        t += '\\hline\n'
        t += fr('Reference EIS fit, $M{=}14$', f['eis_reference_dense']) + fr('Reference EIS fit, $M{=}6$', f['eis_reference'])
        t += '\\hline\n'
        t += fr('Pulse test, $M{=}14$', f['pulse_test_dense']) + fr('Pulse test, $M{=}6$', f['pulse_test'])
        t += '\\hline\n'
        t += row('RLS, 1 RC$^{\\dagger}$', '--', X['rls_order_1']) + row('RLS, 2 RC$^{\\dagger}$', '--', X['rls_order_2'])
        t += fr('Batch fit, $M{=}14^{\\dagger}$', X['batch_fit_dense']) + fr('Batch fit, $M{=}6^{\\dagger}$', X['batch_fit'])
        (out / 'tables' / 'baselines_paper.tex').write_text(t + '\\hline\n\\end{tabular}\n')
    (out / 'RESULTS.md').write_text('# Results\n\nBackend: ' + ', '.join(S['backend']) + '. Versions: ' + json.dumps(S['versions']) + '\n\n' + '\n'.join(L))
    print(f'summary.json, RESULTS.md and tables written to {out}' + (f' ({len(missing)} tasks missing)' if missing else ''))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
