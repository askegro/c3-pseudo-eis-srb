from .experiments import load_config

def build_main(cfg):
    T = []
    nom = cfg['nominal']
    ag = cfg['aging']
    for temp in cfg['grid']['temp_c']:
        for soc in cfg['grid']['soc']:
            is_nom = abs(soc - nom['soc']) < 1e-09 and abs(temp - nom['temp_c']) < 1e-09
            is_ag = abs(soc - ag['soc']) < 1e-09 and abs(temp - ag['temp_c']) < 1e-09
            T.append(dict(kind='opoint', group='map', soc=soc, temp_c=temp, soh=1.0, monte_carlo=cfg['monte_carlo_runs'] if is_nom or is_ag else 0))
    for soh in ag['soh']:
        T.append(dict(kind='opoint', group='aging', soc=ag['soc'], temp_c=ag['temp_c'], soh=soh, monte_carlo=0))
    for amp in cfg['amplitudes_a']:
        T.append(dict(kind='opoint', group='amplitude', soc=nom['soc'], temp_c=nom['temp_c'], soh=1.0, load_current_a=amp, monte_carlo=0))
    for variant in ('wltc', 'plain'):
        T.append(dict(kind='drive', group='drive', variant=variant, soc=nom['soc'], temp_c=nom['temp_c']))
    for case in cfg['pack']['cases']:
        T.append(dict(kind='pack', group='pack', case=case, soc=nom['soc'], temp_c=nom['temp_c']))
    for k, t in enumerate(T):
        t['id'] = k
        t['name'] = '{:03d}_{}_{}'.format(k, t['group'], '_'.join((f'{a}{t[b]}' for a, b in (('z', 'soc'), ('T', 'temp_c'), ('h', 'soh'), ('I', 'load_current_a'), ('', 'variant'), ('', 'case')) if b in t)))
    return T

def build_extra(cfg):
    T = []
    nom = cfg['nominal']
    for temp in cfg['grid']['temp_c']:
        for soc in cfg['grid']['soc']:
            T.append(dict(kind='xmap', group='xmap', soc=soc, temp_c=temp))
    for temp in cfg['extra']['drive_temps_c']:
        T.append(dict(kind='xdrive', group='xdrive', soc=nom['soc'], temp_c=temp))
    for k, t in enumerate(T):
        t['id'] = k
        t['name'] = '{:03d}_{}_z{}_T{}'.format(k, t['group'], t['soc'], t['temp_c'])
    return T
STUDIES = ('main', 'extra')

def tasks(cfg, study):
    return {'main': build_main, 'extra': build_extra}[study](cfg)
