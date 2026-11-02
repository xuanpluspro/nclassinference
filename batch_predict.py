"""Run model checkpoints sequentially in independent CUDA processes."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',required=True,type=Path,help='JSON list of {name,weights,model_config?}')
    p.add_argument('--input',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    args,extra=p.parse_known_args()
    # These are owned by the batch runner, never silently override child paths.
    if any(x.split('=')[0] in {'--weights','--output','--input','--model-config'} for x in extra): p.error('Set model weights/config in experiments JSON')
    experiments=json.loads(args.experiments.read_text(encoding='utf-8-sig'))
    if not isinstance(experiments,list) or not experiments: p.error('Expected nonempty experiment list')
    seen=set(); base=args.experiments.resolve().parent
    for e in experiments:
        name=e['name']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or name.casefold() in seen: p.error('Names must be unique safe directory names')
        seen.add(name.casefold())
        for field in ('weights','model_config'):
            if field in e:
                path=Path(e[field]); path=path if path.is_absolute() else base/path
                if not path.is_file(): raise FileNotFoundError(path)
    for e in experiments:
        weight=Path(e['weights']); weight=weight if weight.is_absolute() else base/weight
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('predict.py')),'--weights',str(weight),'--input',str(args.input),'--output',str(args.output/e['name'])]
        if 'model_config' in e:
            cfg=Path(e['model_config']); cmd+=['--model-config',str(cfg if cfg.is_absolute() else base/cfg)]
        print('\nRunning: '+e['name'],flush=True)
        subprocess.run(cmd+extra,check=True)
    print('All models finished. Use compare.py with these result directories.',flush=True)

if __name__=='__main__': main()
