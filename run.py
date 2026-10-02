import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
USAGE = 'usage: python run.py <command> [options]\n\n  quick              run everything on a fast linear surrogate (about 2 minutes), then report\n  full [--jobs N]    run all DFN simulations of the paper (hours), then report\n  report [NAME]      summaries, tables and figures for results/NAME (default: full)\n                     or for the stored paper results with:  python run.py report reference\n  test               numerical checks of the theory and of the code\n'

def base(name):
    return ROOT / 'reference' if name == 'reference' else ROOT / 'results' / name

def execute(study, config, out, jobs=None, ids=()):
    from prbsid import execute as ex
    argv = [study, *map(str, ids), '--config', str(config), '--out', str(out)]
    if jobs:
        argv += ['--jobs', str(jobs)]
    return ex.main(argv)

def report(name, config):
    from prbsid import summarize, summarize_extra, plots
    root = base(name)
    main_tasks, extra_tasks = (root / 'main' / 'tasks', root / 'extra' / 'tasks')
    if not main_tasks.exists() and (not extra_tasks.exists()):
        print(f'no results found in {root}')
        return 1
    code = 0
    if main_tasks.exists():
        code |= summarize.main(['--tasks', str(main_tasks), '--config', str(config)])
        code |= plots.main(['--tasks', str(main_tasks), '--config', str(config)])
    if extra_tasks.exists():
        code |= summarize_extra.main(['--tasks', str(extra_tasks), '--config', str(config)])
    print(f'summaries, tables and figures are in {root}')
    return code

def main(argv):
    if not argv or argv[0] not in ('quick', 'full', 'report', 'test'):
        print(USAGE)
        return 1
    cmd, rest = (argv[0], argv[1:])
    jobs = int(rest[rest.index('--jobs') + 1]) if '--jobs' in rest else 2
    if cmd == 'test':
        return subprocess.call([sys.executable, str(ROOT / 'tests' / 'test_theory.py')])
    if cmd == 'report':
        name = rest[0] if rest and (not rest[0].startswith('--')) else 'full'
        return report(name, ROOT / ('config_quick.json' if name == 'quick' else 'config.json'))
    config = ROOT / ('config_quick.json' if cmd == 'quick' else 'config.json')
    if cmd == 'quick':
        code = subprocess.call([sys.executable, str(ROOT / 'tests' / 'test_theory.py')])
        if code:
            return code
        jobs = max(jobs, 4)
    for study in ('main', 'extra'):
        if execute(study, config, base(cmd) / study / 'tasks', jobs=jobs):
            print('some tasks failed; see the messages above')
            return 1
    return report(cmd, config)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
