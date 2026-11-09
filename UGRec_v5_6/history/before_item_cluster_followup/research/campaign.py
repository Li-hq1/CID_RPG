"""One dataset, two fixed methods, one seed. No hyperparameter search."""

import argparse
import json
import os
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT, canonical_category


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category",required=True,choices=["Beauty","sports","toys"])
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    category=canonical_category(args.category)
    base=ROOT/"outputs/fm_ablation"/category
    configs=base/"configs"
    configs.mkdir(parents=True,exist_ok=True)
    path=base/"status.json"
    if path.exists():raise FileExistsError(path)
    status={"category":category,"device":args.device,"seed":2024,"state":"running",
            "started_unix":time.time(),"runs":[]}
    write_json(path,status)
    for name in ["control","direct_ce"]:
        config=json.loads((ROOT/f"configs/{name}.json").read_text())
        run_id=f"{category.lower()}_{name}_s2024"
        config["data"]["category"]=category
        config["training"].update(run_id=run_id,device=args.device,seed=2024,epochs=40,
            max_steps=None,smoke_test=False,output_dir="outputs/fm_ablation/runs")
        target=configs/f"{run_id}.json"
        log_path=base/f"{run_id}.log"
        if target.exists() or log_path.exists():raise FileExistsError(run_id)
        write_json(target,config)
        entry={"name":name,"state":"running","started_unix":time.time()}
        status["runs"].append(entry);write_json(path,status)
        print(f"START {category} {name}",flush=True)
        with log_path.open("x") as log:
            completed=subprocess.run([sys.executable,str(ROOT/"run.py"),"--config",str(target)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"})
        entry.update(returncode=completed.returncode,state="complete" if completed.returncode==0 else "failed")
        if completed.returncode:
            status["state"]="failed";write_json(path,status);raise SystemExit(completed.returncode)
        directory=ROOT/"outputs/fm_ablation/runs"/category/run_id
        entry["results"]=json.loads((directory/"results.json").read_text())
        write_json(path,status)
        print(f"DONE {category} {name}: {entry['results']['validation']}",flush=True)
    for name in ["control", "direct_ce"]:
        directory=ROOT/"outputs/fm_ablation/runs"/category/f"{category.lower()}_{name}_s2024"
        with (base/f"{name}_inference.log").open("x") as log:
            completed=subprocess.run([sys.executable,str(ROOT/"research/measure_inference.py"),
                "--run",str(directory),"--device",args.device],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            status["state"]="inference_check_failed";write_json(path,status)
            raise SystemExit(completed.returncode)
    status.update(state="complete",finished_unix=time.time());write_json(path,status)


if __name__=="__main__":main()
