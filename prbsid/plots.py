import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from . import fit as ft, sequences as sq
from .experiments import load_config
from .tasks import tasks as build_tasks
BLUE, ORANGE, VIOLET, INK, MUTED = ('#2a78d6', '#eb6834', '#4a3aa7', '#0b0b0b', '#52514e')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'axes.grid': True, 'grid.color': '#d9d8d4', 'grid.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK})
W = 3.3

def load(tasks_dir, tasks):
    R = {}
    for t in tasks:
        fj = tasks_dir / f"{t['name']}.json"
        if fj.exists():
            r = json.loads(fj.read_text())
            if r.get('status') == 'complete':
                r['_z'] = np.load(tasks_dir / f"{t['name']}.npz")
                R[t['name']] = r
    return R

def thin(n, k):
    return np.unique(np.round(np.linspace(0, n - 1, min(n, k))).astype(int))

def fig_noise(r, cfg, out):
    z = r['_z']
    taus = np.asarray(cfg['fit']['taus_s'])
    names = [b['name'] for b in cfg['bands']]
    cols = [BLUE, ORANGE, VIOLET]
    mk = ['o', 's', '^']
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 1.9), gridspec_kw=dict(wspace=0.28))
    fd = np.logspace(np.log10(z['f'].min()), np.log10(z['f'].max()), 400)
    o = np.argsort(z['f'])
    ax[0].plot(1000.0 * z['zref'].real[o], -1000.0 * z['zref'].imag[o], color=MUTED, lw=1.6, label='reference EIS')
    for k, nme in enumerate(names):
        sel = np.flatnonzero(z['band'] == k)
        sel = sel[thin(sel.size, 28)]
        ax[0].errorbar(1000.0 * z['z'].real[sel], -1000.0 * z['z'].imag[sel], xerr=2000.0 * z['sigma'][sel], yerr=2000.0 * z['sigma'][sel], fmt=mk[k], color=cols[k], ms=3, mec='white', mew=0.4, elinewidth=0.5, label='estimate, ' + (f"{cfg['bands'][k]['fc'] / 1000.0:g} kHz" if cfg['bands'][k]['fc'] >= 1000.0 else f"{cfg['bands'][k]['fc']:g} Hz") + ' clock')
    zm = ft.model_impedance(z['theta_dense'], 2 * np.pi * fd, np.asarray(cfg['fit']['taus_dense_s']))
    ax[0].plot(1000.0 * zm.real, -1000.0 * zm.imag, color=INK, lw=0.9, ls='--', label='fit (19), half-decade grid')
    zm = ft.model_impedance(z['theta'], 2 * np.pi * fd, taus)
    ax[0].plot(1000.0 * zm.real, -1000.0 * zm.imag, color=INK, lw=0.7, ls=':', label='fit (19), decade grid')
    ax[0].set_xlabel('Re $Z$ (m$\\Omega$)')
    ax[0].set_ylabel('$-$Im $Z$ (m$\\Omega$)')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6, columnspacing=0.9, borderaxespad=0.2)
    if 'mc_emp_sd' in z.files:
        for k, nme in enumerate(names):
            sel = np.flatnonzero(z['band'] == k)
            sub = sel[thin(sel.size, 22)]
            ax[1].loglog(z['f'][sel], 1000000.0 * z['mc_emp_sd'][sel], color=cols[k], lw=1.0, label='empirical' if k == 0 else None)
            ax[1].loglog(z['f'][sub], 1000000.0 * z['mc_pred_sd'][sub], mk[k], color=cols[k], ms=3, mec='white', mew=0.4, label='predicted (16)' if k == 0 else None)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Std. dev. of Re $\\hat Z$, Im $\\hat Z$ ($\\mu\\Omega$)')
        ax[1].legend(loc='best')
    else:
        ax[1].set_visible(False)
    fig.savefig(out / 'fig_noise.pdf')
    plt.close(fig)

def fig_maps(R, tasks, cfg, out):
    temps = cfg['grid']['temp_c']
    fig, ax = plt.subplots(1, len(temps), figsize=(W, 1.7), sharex=True, gridspec_kw=dict(wspace=0.62))
    for a, T in zip(np.atleast_1d(ax), temps):
        rs = sorted([R[t['name']] for t in tasks if t['group'] == 'map' and t['temp_c'] == T and (t['name'] in R)], key=lambda r: r['task']['soc'])
        if not rs:
            continue
        soc = np.array([100 * r['task']['soc'] for r in rs])
        g = lambda k, q='r_pulse': 1000.0 * np.array([r['fits'][k][q] for r in rs])
        a.plot(soc, g('eis_reference_dense'), color=MUTED, lw=1.6, label='reference EIS')
        a.errorbar(soc, g('proposed_dense'), yerr=2 * g('proposed_dense', 'r_pulse_sd'), fmt='o', color=BLUE, ms=3, mec='white', mew=0.4, elinewidth=0.6, label='proposed')
        a.plot(soc, g('pulse_test_dense'), 's', color=ORANGE, ms=3, mec='white', mew=0.4, label='pulse test')
        a.set_title(f'{T:g} °C', fontsize=8, pad=3)
        a.set_xlabel('SOC (%)')
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    np.atleast_1d(ax)[0].set_ylabel('$R_{10}$ (m$\\Omega$)')
    h, l = np.atleast_1d(ax)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.12), handlelength=1.4, columnspacing=1.0)
    fig.savefig(out / 'fig_maps.pdf')
    plt.close(fig)

def fig_aging(S, out):
    rows = S.get('aging')
    if not rows:
        return
    soh = np.array([100 * r['soh'] for r in rows])
    g = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)
    h = lambda k, v: np.array([np.nan if r[k][v] is None else r[k][v] for r in rows], float)
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True, gridspec_kw=dict(hspace=0.18))
    ax[0].plot(soh, g('r_pulse_eis_mohm'), color=MUTED, lw=1.6, label='reference EIS')
    ax[0].errorbar(soh, g('r_pulse_mohm'), yerr=2 * g('r_pulse_sd_mohm'), fmt='o', color=BLUE, ms=3.5, mec='white', mew=0.4, elinewidth=0.6, label='proposed')
    ax[0].set_ylabel('$R_{10}$ (m$\\Omega$)')
    ax[0].legend(loc='best')
    ax[1].plot(soh, h('rmse_bol_model_mv', 'wltc'), 's-', color=ORANGE, label='beginning-of-life model')
    ax[1].plot(soh, h('rmse_reidentified_mv', 'wltc'), 'o-', color=BLUE, label='re-identified')
    ax[1].set_ylabel('Voltage error, WLTC (mV)')
    ax[1].set_xlabel('State of health (%)')
    ax[1].legend(loc='best')
    ax[1].set_xticks(soh)
    ax[1].set_xticklabels([f'{x:g}' for x in soh])
    ax[1].invert_xaxis()
    fig.savefig(out / 'fig_aging.pdf')
    plt.close(fig)

def fig_amplitude(S, out):
    rows = S.get('amplitude')
    if not rows:
        return
    a = np.array([r['load_a'] for r in rows])
    fig, ax = plt.subplots(figsize=(W, 1.7))
    ax.loglog(a, [100 * r['mean_rel_err'] for r in rows], 'o-', color=BLUE, label='with measurement noise')
    ax.loglog(a, [100 * r['noise_free_mean_rel_err'] for r in rows], 's--', color=ORANGE, label='noise-free')
    ax.set_xlabel('Load current $I_0$ (A)')
    ax.set_ylabel('Mean impedance error (%)')
    ax.legend(loc='best')
    ax.set_xticks(a)
    ax.set_xticklabels([f'{x:g}' for x in a])
    ax.minorticks_off()
    fig.savefig(out / 'fig_amplitude.pdf')
    plt.close(fig)

def line_power(i, period_samples, periods):
    n = period_samples * periods
    X = np.fft.rfft(i[:n]) / n
    p = 2 * np.abs(X) ** 2
    lmax = sq.n_lines(period_samples // 4) if False else (len(p) - 1) // periods - 1
    l = np.arange(1, lmax)
    half = periods // 2
    return (l, np.array([p[k * periods - half:k * periods + half + periods % 2].sum() for k in l]))

def fig_wltc(R, tasks, cfg, out):
    nom = cfg['nominal']
    rc = next((R[t['name']] for t in tasks if t['group'] == 'map' and abs(t['soc'] - nom['soc']) < 1e-09 and (abs(t['temp_c'] - nom['temp_c']) < 1e-09) and (t['name'] in R)), None)
    rw = next((R[t['name']] for t in tasks if t['group'] == 'drive' and t.get('variant') == 'wltc' and (t['name'] in R)), None)
    if rc is None or rw is None:
        return
    b0 = cfg['bands'][0]
    N = 2 ** cfg['sequence_degree'] - 1
    Lp = N * cfg['kappa']
    P = b0['periods']
    lm = sq.n_lines(N)
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 1.75), gridspec_kw=dict(wspace=0.28))
    for r, col, lab, st in ((rc, BLUE, 'constant load', '-'), (rw, ORANGE, 'WLTC load', '--')):
        key = 'i_slow'
        l, p = line_power(r['_z'][key], Lp, P)
        sel = l <= lm
        ax[0].loglog(l[sel] * b0['fc'] / N, p[sel], st, color=col, lw=1.0, label=lab)
    I = cfg['load_current_a']
    l = np.arange(1, lm + 1)
    ax[0].loglog(l * b0['fc'] / N, 2 * (I * sq.line_magnitude(l, N)) ** 2, ':', color=INK, lw=0.9, label='Lemma 1, $2I_0^2|c_\\ell|^2$')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Input power per line (A$^2$)')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.5, columnspacing=0.8, borderaxespad=0.2)
    for r, col, lab, mkr in ((rc, BLUE, 'constant load', 'o'), (rw, ORANGE, 'WLTC load', 's')):
        z = r['_z']
        e = 100 * np.abs(z['z'] - z['zref']) / np.abs(z['zref'])
        o = np.argsort(z['f'])
        f = z['f'][o]
        e = e[o]
        edges = np.logspace(np.log10(f.min()), np.log10(f.max()), 25)
        c = np.sqrt(edges[1:] * edges[:-1])
        idx = np.digitize(f, edges) - 1
        med = np.array([np.median(e[idx == k]) if np.any(idx == k) else np.nan for k in range(len(c))])
        ax[1].loglog(c, med, mkr + '-', color=col, ms=3, mec='white', mew=0.4, label=lab)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Median impedance error (%)')
    fig.savefig(out / 'fig_wltc.pdf')
    plt.close(fig)

def fig_pack(R, tasks, cfg, out):
    r = next((R[t['name']] for t in tasks if t['group'] == 'pack' and t.get('case') == 'both' and (t['name'] in R)), None) or next((R[t['name']] for t in tasks if t['group'] == 'pack' and t['name'] in R), None)
    if r is None or 'pack_t' not in r['_z'].files:
        return
    z = r['_z']
    band = next((b for b in cfg['bands'] if b['name'] == 'mid'))
    n = 60 * band['os_sim']
    t = 1000.0 * (z['pack_t'][-n:] - z['pack_t'][-n])
    s = z['pack_s'][-n:]
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True, gridspec_kw=dict(hspace=0.25))
    ax[0].plot(t, s * z['pack_v_p'][-n:], color=ORANGE, lw=0.9)
    ax[0].set_ylabel('Single group (V)')
    ax[0].set_title('Contribution to the pack voltage', fontsize=8, pad=3)
    ax[1].plot(t, z['pack_v_ins'][-n:], color=BLUE, lw=0.9)
    ax[1].set_ylabel('Complementary pair (V)')
    ax[1].set_xlabel('Time (ms)')
    ax[1].ticklabel_format(axis='y', useOffset=False)
    fig.savefig(out / 'fig_pack.pdf')
    plt.close(fig)

def main(argv):
    opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    tasks_dir = Path(opt('--tasks'))
    out = Path(opt('--out', tasks_dir.parent / 'figures'))
    cfg = load_config(opt('--config'))
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(cfg, 'main')
    R = load(tasks_dir, tasks)
    sfile = tasks_dir.parent / 'summary.json'
    S = json.loads(sfile.read_text()) if sfile.exists() else {}
    nom = cfg['nominal']
    rn = next((R[t['name']] for t in tasks if t['group'] == 'map' and abs(t['soc'] - nom['soc']) < 1e-09 and (abs(t['temp_c'] - nom['temp_c']) < 1e-09) and (t['name'] in R)), None)
    made = []
    for name, fn in (('fig_noise', lambda: fig_noise(rn, cfg, out) if rn else None), ('fig_maps', lambda: fig_maps(R, tasks, cfg, out)), ('fig_aging', lambda: fig_aging(S, out)), ('fig_amplitude', lambda: fig_amplitude(S, out)), ('fig_wltc', lambda: fig_wltc(R, tasks, cfg, out)), ('fig_pack', lambda: fig_pack(R, tasks, cfg, out))):
        fn()
        if (out / f'{name}.pdf').exists():
            made.append(name)
    print('figures written to', out, ':', ', '.join(made))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
