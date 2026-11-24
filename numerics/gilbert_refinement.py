"""Higher-budget Rust Gilbert comparison, with raw preselection retained.

Eight independent subprocesses run concurrently by default. The analytical
distance is used ONLY as an optional benchmark stopping reference, never to
initialize, propose or weight any iterate. Set --gap 0 to disable that stop.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import csv
import gzip
import json
import shutil
import subprocess
import numpy as np
from gilbert_comparison import HERE, RUST, RAW, run_one, filehash, UPSTREAM, COMMIT
from rank2_validation import SQRT2


def compress_raw(result):
    run=RAW/result['raw_directory']
    for name in ['input_distances.csv','stdout.log','stderr.log']:
        path=run/name
        if path.exists():
            with path.open('rb') as source, gzip.open(str(path)+'.gz','wb',compresslevel=6) as dest:
                shutil.copyfileobj(source,dest)
            path.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--corrections',type=int,default=100000)
    p.add_argument('--trials',type=int,default=100000000)
    p.add_argument('--sweeps',type=int,default=2000)
    p.add_argument('--gradient-steps',type=int,default=3)
    p.add_argument('--gap',type=float,default=1e-8)
    args=p.parse_args()
    assert args.workers>=1 and args.gap>=0
    if args.binary:binary=args.binary.resolve()
    else:
        build=HERE/'.gilbert-build'
        subprocess.run(['cargo','build','--release','--locked','--target-dir',str(build)],cwd=RUST,check=True)
        binary=build/'release/rusty-code'
    baseline=list(csv.DictReader((HERE/'rank2-validation.csv').open()))
    jobs=[(float(r['alpha']),42) for r in baseline]+[(1/SQRT2,s) for s in [43,44]]
    def task(job):
        result=run_one(job[0],job[1],args.corrections,binary,
            trial_limit=args.trials,preselection=True,gradient_steps=args.gradient_steps,
            local_sweeps=args.sweeps,reference_gap=args.gap or None)
        result=dict(result,trial_limit=args.trials,preselection=True,
                    local_sweeps=args.sweeps,gradient_steps=args.gradient_steps,
                    reference_gap=args.gap,optimizer='grad')
        compress_raw(result)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(task,j):j for j in jobs}
        for future in as_completed(pending):
            results.append(future.result())
            print(f'Completed {len(results)}/{len(jobs)}',flush=True)
    results.sort(key=lambda r:(r['seed'],r['alpha']))
    with (HERE/'gilbert-refinement.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    primary=[r for r in results if r['seed']==42]
    bell=[r for r in results if abs(r['alpha']-1/SQRT2)<1e-12]
    summary=dict(upstream=UPSTREAM,upstream_commit=COMMIT,workers=args.workers,
        primary_samples=len(primary),total_runs=len(results),seed=42,
        correction_budget=args.corrections,trial_limit=args.trials,
        local_sweeps=args.sweeps,gradient_steps=args.gradient_steps,learning_rate=.3,
        preselection=True,reference_gap=args.gap,
        reference_use='Stopping only; no reference values enter trial generation or updates.',
        max_distance2_excess=max(r['distance2_excess'] for r in primary),
        max_distance_excess=max(r['distance_excess'] for r in primary),
        reference_gap_reached=sum(r['stop_reason']=='reference_gap' for r in results),
        trial_limited=sum(r['stop_reason']=='trial_limit' for r in results),
        correction_limited=sum(r['stop_reason']=='correction_budget' for r in results),
        min_entangled_corrections=min(r['corrections'] for r in results if r['alpha']>0),
        max_corrections=max(r['corrections'] for r in results),
        max_trials_used=max(r['trials'] for r in results),
        max_trace_error=max(r['trace_error'] for r in results),
        min_eig=min(r['min_eig'] for r in results),
        min_pt_eig=min(r['min_pt_eig'] for r in results),
        metric_recheck_max_error=max(r['metric_recheck_error'] for r in results),
        bell_results=bell,
        source_sha256={str(f.relative_to(RUST)):filehash(f) for f in sorted(RUST.glob('src/*.rs'))},
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip())
    (HERE/'gilbert-refinement-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
