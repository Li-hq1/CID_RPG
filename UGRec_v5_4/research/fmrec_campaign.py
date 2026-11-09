"""A bounded Beauty screen: original FMRec, no FM loss, or history-only CE."""

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
    parser.add_argument('--method',choices=['fmrec','ce_only','history_ce'],required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    base=ROOT/'outputs/fmrec';base.mkdir(parents=True,exist_ok=True)
    status_path=base/f'status_{args.method}.json'
    if status_path.exists():raise FileExistsError(status_path)
    name={'fmrec':'fmrec','ce_only':'fmrec_ce_only','history_ce':'fmrec_history_ce'}[args.method]
    config=json.loads((ROOT/f'configs/{name}.json').read_text())
    config['training'].update(device=args.device,epochs=40,eval_interval=5,patience=8,
                              run_id=f'beauty_{args.method}_initial40_s2024',seed=2024,
                              max_steps=None,smoke_test=False)
    configs=base/'configs';configs.mkdir(exist_ok=True)
    config_path=configs/f'{args.method}_initial40.json'
    if config_path.exists():raise FileExistsError(config_path)
    write_json(config_path,config)
    status={'method':args.method,'device':args.device,'state':'running','started_unix':time.time()}
    write_json(status_path,status)
    log=base/f'{args.method}_initial40.log'
    command=[sys.executable,str(ROOT/'run_fmrec.py'),'--config',str(config_path)]
    with log.open('x') as stream:
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
