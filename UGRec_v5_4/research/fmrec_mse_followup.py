"""Run the single predeclared MSE candidate only if the budget extension fails."""

import argparse
import json
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    base=ROOT/'outputs/fmrec/optimization'
    status_path=base/'status_mse_followup.json'
    if status_path.exists():raise FileExistsError(status_path)
    status={'state':'waiting_first_stage','device':args.device,'started_unix':time.time(),
        'criterion':'FMRec selection NDCG10 <= matched history CE selection NDCG10 after extension'}
    write_json(status_path,status)
    while True:
        stages=[json.loads((base/f'status_{method}.json').read_text()) for method in ['fmrec','history_ce']]
        if any(stage['state']=='failed' for stage in stages):
            status.update(state='dependency_failed',finished_unix=time.time());write_json(status_path,status)
            raise SystemExit('First-stage dependency failed; candidate not launched.')
        if all(stage['state']=='complete' for stage in stages):break
        time.sleep(15)
    fmrec,history=[stage['results']['validation_selection']['ndcg@10'] for stage in stages]
    status['first_stage_selection_ndcg10']={'fmrec':fmrec,'history_ce':history}
    if fmrec>history:
        status.update(state='skipped',reason='FMRec already exceeds history baseline on selection',finished_unix=time.time())
        write_json(status_path,status);print(json.dumps(status),flush=True);return
    status.update(state='running_single_mse_candidate');write_json(status_path,status)
    process=subprocess.run([sys.executable,str(ROOT/'research/fmrec_optimization.py'),
        '--method','fmrec_mse','--device',args.device],cwd=ROOT)
    status.update(state='complete' if process.returncode==0 else 'failed',returncode=process.returncode,finished_unix=time.time())
    write_json(status_path,status);print(json.dumps(status),flush=True)
    if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
