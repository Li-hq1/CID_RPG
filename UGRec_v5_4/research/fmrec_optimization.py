"""Bounded optimization: extend matched trajectories, then one history-MSE trial."""

import argparse
import json
import os
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['fmrec','ce_only','history_ce','fmrec_mse'],required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    base=ROOT/'outputs/fmrec/optimization';base.mkdir(parents=True,exist_ok=True)
    status_path=base/f'status_{args.method}.json'
    if status_path.exists():raise FileExistsError(status_path)
    method='fmrec' if args.method=='fmrec_mse' else args.method
    name={'fmrec':'fmrec','ce_only':'fmrec_ce_only','history_ce':'fmrec_history_ce'}[method]
    config=json.loads((ROOT/f'configs/{name}.json').read_text())
    variant='mse140' if args.method=='fmrec_mse' else 'extend140'
    config['training'].update(device=args.device,epochs=140,eval_interval=5,patience=100,
        run_id=f'beauty_{method}_{variant}_s2024',seed=2024,max_steps=None,smoke_test=False)
    if args.method=='fmrec_mse':config['fmrec']['mse_weight']=.4
    configs=base/'configs';configs.mkdir(exist_ok=True)
    config_path=configs/f'{args.method}.json'
    if config_path.exists():raise FileExistsError(config_path)
    write_json(config_path,config)
    command=[sys.executable,str(ROOT/'run_fmrec.py'),'--config',str(config_path)]
    if args.method!='fmrec_mse':
        parent=ROOT/f'outputs/fmrec/runs/Beauty/beauty_{method}_initial40_s2024/best.pt'
        command+=['--resume-from',str(parent)]
    status={'method':args.method,'device':args.device,'state':'running','started_unix':time.time(),
        'command':command,'predeclared_plan':'docs/FMRec优化实验.md'}
    write_json(status_path,status)
    with (base/f'{args.method}.log').open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    status.update(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode,
                  finished_unix=time.time())
    if result.returncode==0:
        directory=ROOT/config['training']['output_dir']/'Beauty'/config['training']['run_id']
        status['results']=json.loads((directory/'results.json').read_text())
    write_json(status_path,status);print(json.dumps(status),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
