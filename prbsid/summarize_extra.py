import json, sys
from pathlib import Path
import numpy as np
from .experiments import load_config
from .tasks import tasks as build_tasks

def main(argv):
    tdir = Path(argv[argv.index('--tasks') + 1])
    cfg = load_config(argv[argv.index('--config') + 1] if '--config' in argv else None)
    tasks = build_tasks(cfg, 'extra')
    R, missing = ({}, [])
    for t in tasks:
        f = tdir / f"{t['name']}.json"
        d = json.loads(f.read_text()) if f.exists() else None
        if d and d.get('status') == 'complete':
            R[t['name']] = d
        else:
            missing.append((t['name'], d.get('status') if d else 'no result'))
    if missing and '--allow-missing' not in argv:
        print('Missing or incomplete tasks:')
        [print('  ', n, '-', s) for n, s in missing]
        return 1
    xm = [R[t['name']] for t in tasks if t['group'] == 'xmap' and t['name'] in R]
    xd = [R[t['name']] for t in tasks if t['group'] == 'xdrive' and t['name'] in R]
    S = dict(missing=missing, n_map=len(xm), n_drive=len(xd))
    L = ['# Extension study', f'map points complete: {len(xm)} / 24, drive points complete: {len(xd)} / 3', '']
    q = lambda a, p: float(np.percentile(a, p))
    if xm:
        r = lambda sel: np.array([sel(x) for x in xm if sel(x) is not None], float)
        S['map'] = {}
        L += ['## PRBS test at every map point (half-decade grid, R10, Monte Carlo)', '| quantity | median | min | max |', '|---|---|---|---|']
        for lab, sel in (('bias vs reference (%)', lambda x: x['prbs']['r10_half_decade']['bias_pct']), ('sd empirical (% of R10)', lambda x: x['prbs']['r10_half_decade']['sd_emp_pct']), ('sd predicted / empirical', lambda x: x['prbs']['r10_half_decade']['sd_ratio_pred_over_emp']), ('coverage of 2 sd vs reference', lambda x: x['prbs']['r10_half_decade']['coverage_2sd_vs_ref']), ('R_tot (decade) sd predicted / empirical', lambda x: x['prbs']['r_tot_decade_sd_ratio']), ('R10 (decade) sd predicted / empirical', lambda x: x['prbs']['r10_decade']['sd_ratio_pred_over_emp'])):
            a = r(sel)
            S['map'][lab] = dict(median=q(a, 50), min=float(a.min()), max=float(a.max()))
            L.append(f'| {lab} | {q(a, 50):.3f} | {a.min():.3f} | {a.max():.3f} |')
        L += ['', f"PRBS test duration: {xm[0]['prbs']['duration_s']:.0f} s", '', '## Pulse tests at every map point (half-decade grid, R10)', '| protocol | duration (s) | median bias (%) | median sd (%) | median sd pred/emp | median coverage 2 sd | median RMSE to ref (%) |', '|---|---|---|---|---|---|---|']
        names = [p for p in xm[0]['pulse']]
        S['pulse'] = {}
        for n in names:
            ok = [x['pulse'][n] for x in xm if x['pulse'].get(n, {}).get('complete')]
            if not ok:
                continue
            g = lambda k: float(np.median([o['half_decade']['r10'][k] for o in ok]))
            S['pulse'][n] = dict(n=len(ok), duration_s=ok[0]['duration_s'], bias_pct=g('bias_pct'), sd_emp_pct=g('sd_emp_pct'), ratio=g('sd_ratio_pred_over_emp'), coverage=g('coverage_2sd_vs_ref'), rmse_pct=g('rmse_to_ref_pct'))
            s = S['pulse'][n]
            L.append(f"| {n} ({len(ok)} pts) | {s['duration_s']:.0f} | {s['bias_pct']:.2f} | {s['sd_emp_pct']:.3f} | {s['ratio']:.2f} | {s['coverage']:.2f} | {s['rmse_pct']:.2f} |")
        L += ['', '## Validation RMSE (mV), median over map points, half-decade grid']
        for rec in ('pulse_load', 'pulse_two_c', 'wltc'):
            row = {'prbs': float(np.nanmedian([1000.0 * x['prbs']['validation_rmse_v_half_decade'][rec] for x in xm if x['prbs']['validation_rmse_v_half_decade'].get(rec) is not None]))}
            for n in names:
                v = [1000.0 * x['pulse'][n]['half_decade']['validation_rmse_v_mean'][rec] for x in xm if x['pulse'].get(n, {}).get('complete') and x['pulse'][n]['half_decade']['validation_rmse_v_mean'].get(rec) is not None]
                if v:
                    row[n] = float(np.median(v))
            S.setdefault('validation_mv', {})[rec] = row
            L.append(f'- {rec}: ' + ', '.join((f'{k} {v:.2f}' for k, v in row.items())))
    if xd:
        L += ['', '## Drive cycle: batch fit of the load alone against the PRBS test (R10, half-decade grid)', '| T (C) | method | duration (s) | bias (%) | sd (%) | sd pred/emp | coverage 2 sd | val WLTC RMSE (mV) |', '|---|---|---|---|---|---|---|---|']
        S['drive'] = []
        for x in xd:
            T = x['task']['temp_c']
            for n, b in x['batch'].items():
                if not b.get('complete'):
                    continue
                s = b['half_decade']['r10']
                v = b['half_decade']['validation_rmse_v_mean'].get('wltc')
                L.append(f"| {T:g} | batch {n} | {b['duration_s']:.0f} | {s['bias_pct']:.2f} | {s['sd_emp_pct']:.3f} | {s['sd_ratio_pred_over_emp']:.2f} | {s['coverage_2sd_vs_ref']:.2f} | {1000.0 * v:.2f} |")
                S['drive'].append(dict(temp_c=T, method=n, **s, duration_s=b['duration_s'], val_wltc_mv=1000.0 * v))
            s = x['prbs']['r10_half_decade']
            v = x['prbs']['validation_rmse_v_half_decade'].get('wltc')
            L.append(f"| {T:g} | PRBS under WLTC | {x['prbs']['duration_s']:.0f} | {s['bias_pct']:.2f} | {s['sd_emp_pct']:.3f} | {s['sd_ratio_pred_over_emp']:.2f} | {s['coverage_2sd_vs_ref']:.2f} | {1000.0 * v:.2f} |")
            S['drive'].append(dict(temp_c=T, method='prbs', **s, duration_s=x['prbs']['duration_s'], val_wltc_mv=1000.0 * v))
    if xm:
        temps = sorted({x['task']['temp_c'] for x in xm}, reverse=True)
        hd = lambda x: x['prbs']['r10_half_decade']
        pb = lambda x, n: abs(x['pulse'][n]['half_decade']['r10']['bias_pct'])
        pn = [n for n in xm[0]['pulse'] if all((x['pulse'].get(n, {}).get('complete') for x in xm))]
        L += ['', '## Mean absolute R10 error against the reference EIS by temperature (%), half-decade grid', '| T (C) | pseudo-EIS | ' + ' | '.join(pn) + ' |', '|---|---|' + '---|' * len(pn)]
        S['error_by_temperature'] = {}
        for T in temps:
            sel = [x for x in xm if x['task']['temp_c'] == T]
            row = {'prbs': float(np.mean([abs(hd(x)['bias_pct']) for x in sel]))}
            row.update({n: float(np.mean([pb(x, n) for x in sel])) for n in pn})
            S['error_by_temperature'][f'{T:g}'] = row
            L.append(f"| {T:g} | {row['prbs']:.1f} | " + ' | '.join((f'{row[n]:.1f}' for n in pn)) + ' |')
        sd = lambda f: (float(min((f(x) for x in xm))), float(np.median([f(x) for x in xm])), float(max((f(x) for x in xm))))
        S['sd_ratio_r10_half_decade'] = sd(lambda x: hd(x)['sd_ratio_pred_over_emp'])
        S['sd_ratio_rtot_decade'] = sd(lambda x: x['prbs']['r_tot_decade_sd_ratio'])
        S['bias_over_sd_min'] = float(min((abs(hd(x)['bias_pct']) / hd(x)['sd_emp_pct'] for x in xm)))
        S['sd_pct_range'] = sd(lambda x: hd(x)['sd_emp_pct'])
        S['pulse_smaller_error_points'] = {n: int(sum((pb(x, n) < abs(hd(x)['bias_pct']) for x in xm))) for n in pn}
        S['pulse_sd_ratio_median'] = {n: float(np.median([x['pulse'][n]['half_decade']['r10']['sd_ratio_pred_over_emp'] for x in xm])) for n in pn}
        S['single_vs_repeated_max_diff_pp'] = float(max((abs(pb(x, 'p20_single') - pb(x, 'p20_x21')) for x in xm)))
        L += ['', '## Statistics quoted in the paper', f"- predicted / empirical standard deviation of R10 (half-decade grid): min {S['sd_ratio_r10_half_decade'][0]:.2f}, median {S['sd_ratio_r10_half_decade'][1]:.2f}, max {S['sd_ratio_r10_half_decade'][2]:.2f}", f"- predicted / empirical standard deviation of R_tot (decade grid): min {S['sd_ratio_rtot_decade'][0]:.2f}, median {S['sd_ratio_rtot_decade'][1]:.2f}, max {S['sd_ratio_rtot_decade'][2]:.2f}", f"- empirical standard deviation of R10: {S['sd_pct_range'][0]:.2f} to {S['sd_pct_range'][2]:.2f} % of its value; bias over standard deviation: at least {S['bias_over_sd_min']:.1f}", '- map points at which a pulse test has the smaller R10 error than pseudo-EIS: ' + ', '.join((f'{n} {v} of {len(xm)}' for n, v in S['pulse_smaller_error_points'].items())), '- median predicted / empirical standard deviation of the pulse fit: ' + ', '.join((f'{n} {v:.2f}' for n, v in S['pulse_sd_ratio_median'].items())), f"- largest difference between the single and the repeated 20 A pulse: {S['single_vs_repeated_max_diff_pp']:.2f} percentage points"]
    out = tdir.parent
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    (out / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
