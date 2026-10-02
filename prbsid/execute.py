import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'
import numpy as np
from . import experiments as ex, extra
from .tasks import tasks as build_tasks
ex.RUNNERS.update(extra.RUNNERS)
ROOT = Path(__file__).resolve().parents[1]

def seed_of(cfg, study):
    return cfg['extra']['seed'] if study == 'extra' else cfg['seed']

def run_one(cfg, study, task, out, force=False):
    fj = out / f"{task['name']}.json"
    if fj.exists() and (not force) and (json.loads(fj.read_text()).get('status') == 'complete'):
        print(f"[{task['name']}] already complete, skipped")
        return True
    print(f"[{task['name']}] start, backend={cfg['backend']}", flush=True)
    t0 = time.time()
    rng = np.random.default_rng([seed_of(cfg, study), task['id']])
    try:
        res, arrays = ex.RUNNERS[task['kind']](cfg, task, rng, log=lambda s: print(s, flush=True))
    except Exception:
        res, arrays = (dict(task=task, status='failed', traceback=traceback.format_exc()), {})
        print(res['traceback'], flush=True)
    res['seconds'] = time.time() - t0
    res['backend'] = cfg['backend']
    res['versions'] = dict(numpy=np.__version__, python=sys.version.split()[0])
    try:
        import scipy
        res['versions']['scipy'] = scipy.__version__
        if cfg['backend'] == 'pybamm':
            import pybamm
            res['versions']['pybamm'] = pybamm.__version__
    except Exception:
        pass
    if arrays:
        tmp = out / f"{task['name']}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out / f"{task['name']}.npz")
    tmp = fj.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(ex.jsonable(res), indent=1))
    os.replace(tmp, fj)
    print(f"[{task['name']}] {res['status']} in {res['seconds']:.0f} s", flush=True)
    return res['status'] == 'complete'

def run_ids(study, ids, cfg_path, out, force=False):
    cfg = ex.load_config(cfg_path)
    tasks = build_tasks(cfg, study)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for i in ids or [t['id'] for t in tasks]:
        ok = run_one(cfg, study, tasks[i], out, force) and ok
    return ok

def run_parallel(study, cfg_path, out, jobs):
    cfg = ex.load_config(cfg_path)
    ids = [t['id'] for t in build_tasks(cfg, study)]

    def one(i):
        cmd = [sys.executable, '-B', '-m', 'prbsid.execute', study, str(i), '--config', str(cfg_path), '--out', str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        last = [l for l in r.stdout.splitlines() if l.startswith('[')][-1:] or [r.stderr[-300:]]
        print(last[0], flush=True)
        return r.returncode
    with ThreadPoolExecutor(jobs) as pool:
        codes = list(pool.map(one, ids))
    print(f'{study}: {codes.count(0)} of {len(codes)} tasks complete')
    return all((c == 0 for c in codes))

def main(argv):
    opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    study = argv[0]
    cfg_path = opt('--config', ROOT / 'config.json')
    out = opt('--out', ROOT / 'results' / 'full' / study / 'tasks')
    skip = {study, opt('--config'), opt('--out'), opt('--jobs')}
    ids = [int(a) for a in argv[1:] if not a.startswith('--') and a not in skip]
    if '--jobs' in argv:
        return 0 if run_parallel(study, cfg_path, out, int(opt('--jobs'))) else 1
    return 0 if run_ids(study, ids, cfg_path, out, '--force' in argv) else 1
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
