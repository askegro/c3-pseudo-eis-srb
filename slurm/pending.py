import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prbsid.experiments import load_config
from prbsid.tasks import tasks

def main(argv):
    mode, study, config, out = argv[:4]
    skip = {int(i) for i in argv[4].split()} if len(argv) > 4 else set()
    cfg = load_config(str(ROOT / config))
    out = ROOT / out
    todo = []
    total = 0
    for t in tasks(cfg, study):
        total += 1
        f = out / f"{t['name']}.json"
        state = json.loads(f.read_text()).get('status') if f.exists() else None
        if state != 'complete' and t['id'] not in skip:
            todo.append((t['id'], t['name'], state))
    if mode == 'ids':
        print(','.join(str(i) for i, _, _ in todo))
    else:
        print(f'{study}: complete {total - len(todo)} of {total}')
        for i, name, state in todo:
            print(f'  {i:3d} {name} ({state or "no result"})')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
