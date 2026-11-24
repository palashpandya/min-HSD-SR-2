"""Reproduce the Rust Gilbert comparison, retaining raw states and trajectories.

Run rank2_validation.py first, then this script. By default Cargo builds the
versioned Rust snapshot with its lockfile. Use --binary PATH for an existing
release build. The historical Rust CSV column 'distance' contains SQUARED HSD.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import fcntl
import functools
import inspect
import json
import os
import re
import subprocess
import time

import numpy as np
from scipy.io import mmread
from rank2_validation import ASTAR, SQRT2, distance2, target

HERE = Path(__file__).resolve().parent
RAW = HERE / 'gilbert-runs'
RUST = HERE / 'rusty-code'
BASELINE_RUST = HERE / 'rusty-code-baseline'
UPSTREAM = 'https://github.com/palashpandya/gilbert-opt-rs'
COMMIT = '8df88703d8c5e5cfca2e5b2cb34a2ef962ed3a3f'
TRIAL_LIMIT = 10_000_000
TOL = 1e-15  # A threshold on squared distance, not an optimization-gap tolerance.


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_run(function):
    """Serialize duplicate jobs if two comparison controllers share a cache."""
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        params=inspect.signature(function).bind(*args,**kwargs).arguments
        locks=RAW/'.locks';locks.mkdir(parents=True,exist_ok=True)
        label=f'a{params["alpha"]:.12f}_s{params["seed"]}_c{params["budget"]}'
        with (locks/(label+'.lock')).open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            return function(*args,**kwargs)
    return wrapped


@locked_run
def run_one(alpha, seed, budget, binary, trial_limit=TRIAL_LIMIT,
            optimizer='grad', timeout=None, preselection=True, gradient_steps=3,
            local_sweeps=20, reference_gap=None):
    label = f'a{alpha:.12f}_s{seed}_c{budget}'
    if trial_limit != TRIAL_LIMIT:
        label += f'_t{trial_limit}'
    if optimizer != 'grad':
        label += '_'+optimizer
    if not preselection:
        label += '_nopreselect'
    if gradient_steps != 3:
        label += f'_g{gradient_steps}'
    if local_sweeps != 20:
        label += f'_w{local_sweeps}'
    if reference_gap is not None:
        label += f'_gap{reference_gap:g}'
    run = RAW / label
    run.mkdir(parents=True, exist_ok=True)
    rho = target(alpha)
    rows, cols = np.nonzero(rho)
    entries = [f'{i+1} {j+1} {rho[i,j]:.17g}' for i,j in zip(rows, cols)]
    (run / 'input.mtx').write_text(
        '%%MatrixMarket matrix coordinate real general\n4 4 '
        + str(len(entries)) + '\n' + '\n'.join(entries) + '\n')
    config = f'''[input]
matrix_path = "input.mtx"
N = 2
d = 2
[algorithm]
optimizer = "{optimizer}"
max_corr = {budget}
max_trials = {trial_limit}
tol = {TOL}
seed = {seed}
preselection = {str(preselection).lower()}
gradient_steps = {gradient_steps}
local_sweeps = {local_sweeps}
[output]
dir = "."
'''
    if reference_gap is not None:
        config=config.replace('[output]',f'reference_distance2 = {distance2(alpha):.17g}\n'
                              f'reference_gap = {reference_gap:.17g}\n[output]')
    (run / 'config.toml').write_text(config)
    key = dict(alpha=float(alpha), seed=seed, correction_budget=budget,
               trial_limit=trial_limit, binary_sha256=filehash(binary))
    if optimizer != 'grad':
        key['optimizer'] = optimizer
    if not preselection:
        key['preselection'] = False
    if gradient_steps != 3:
        key['gradient_steps'] = gradient_steps
    if local_sweeps != 20:
        key['local_sweeps'] = local_sweeps
    if reference_gap is not None:
        key['reference_gap'] = reference_gap
    cache = run / 'checked.json'
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get('key') == key:
            return old['result']
    start = time.monotonic()
    with (run/'stdout.log').open('w') as stdout, (run/'stderr.log').open('w') as stderr:
        subprocess.run([str(binary)], cwd=run, text=True, stdout=stdout,
                       stderr=stderr, timeout=timeout, check=True)
    elapsed = time.monotonic() - start
    process_stdout = (run/'stdout.log').read_text()
    history = np.loadtxt(run / 'input_distances.csv', delimiter=',', skiprows=1,
                        ndmin=2)[:, 1]
    matrix = np.asarray(mmread(run / 'input_css.mtx'))
    independent = float(np.sum(np.abs(matrix-rho)**2))
    pt = matrix.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    analytic = distance2(alpha)
    assert abs(independent-history[-1]) < 1e-12
    assert abs(history[0]-2*alpha*alpha*(1-alpha*alpha)) < 1e-12
    assert np.diff(history).max(initial=0.) <= 1e-14
    assert abs(np.trace(matrix)-1) < 1e-11
    assert abs(matrix-matrix.conj().T).max() < 1e-12
    assert np.linalg.eigvalsh(matrix).min() > -1e-11
    assert np.linalg.eigvalsh(pt).min() > -1e-11
    assert independent >= analytic-1e-12
    corrections = len(history)-1
    match = re.search(r'termination: trials=(\d+), corrections=(\d+)',
                      process_stdout)
    trials = int(match[1]) if match else 0
    reason = ('distance_threshold' if history[-1] <= TOL else
              'reference_gap' if reference_gap is not None and
              history[-1]-analytic <= reference_gap+1e-15 else
              'correction_budget' if corrections == budget else 'trial_limit')
    if reason == 'trial_limit':
        assert trials == trial_limit
    result = dict(alpha=float(alpha), seed=seed, correction_budget=budget,
                  corrections=corrections, trials=trials, stop_reason=reason,
                  analytic_distance2=analytic, gilbert_distance2=independent,
                  distance2_excess=independent-analytic,
                  distance_excess=np.sqrt(max(independent,0.))-np.sqrt(analytic),
                  trace_error=float(abs(np.trace(matrix)-1)),
                  min_eig=float(np.linalg.eigvalsh(matrix).min()),
                  min_pt_eig=float(np.linalg.eigvalsh(pt).min()),
                  metric_recheck_error=abs(independent-history[-1]),
                  elapsed_seconds=elapsed, raw_directory=label)
    cache.write_text(json.dumps(dict(key=key,result=result), indent=2)+'\n')
    print(f'{label}: corrections={corrections}, stop={reason}, '
          f'excess_D2={result["distance2_excess"]:.3g}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if args.binary:
        binary = args.binary.resolve()
    else:
        build = HERE / '.gilbert-build'
        subprocess.run(['cargo','build','--release','--locked','--target-dir',str(build)],
                       cwd=BASELINE_RUST, check=True)
        binary = build / 'release/rusty-code'
    baseline = list(csv.DictReader((HERE/'rank2-validation.csv').open()))
    rows = [run_one(float(r['alpha']),42,200,binary) for r in baseline]
    representative = [.01,.05,.2,.3,float(ASTAR),.4,.5,.6,1/SQRT2]
    repeated = [run_one(a,seed,200,binary)
                for a in representative for seed in [43,44]]
    long = [run_one(a,42,2000,binary) for a in representative]
    for a in representative:
        short = min(rows,key=lambda r:abs(r['alpha']-a))
        longer = min(long,key=lambda r:abs(r['alpha']-a))
        assert longer['gilbert_distance2'] <= short['gilbert_distance2']+1e-12
    with (HERE/'gilbert-comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows+repeated+long)
    summary = dict(upstream=UPSTREAM,upstream_commit=COMMIT,
        optimizer='grad', local_sweeps=20, gradient_steps=3, learning_rate=.3,
        initialization='computational-basis diagonal of input',
        preselection=True, distance_threshold_squared=TOL, trial_limit=TRIAL_LIMIT,
        baseline_samples=len(rows), baseline_seed=42, correction_budget=200,
        baseline_max_distance2_excess=max(r['distance2_excess'] for r in rows),
        baseline_max_distance_excess=max(r['distance_excess'] for r in rows),
        baseline_corrections_min=min(r['corrections'] for r in rows if r['alpha']>0),
        baseline_trial_limited=sum(r['stop_reason']=='trial_limit' for r in rows),
        long_budget=2000,long_samples=len(long),long_seed=42,
        long_max_distance2_excess=max(r['distance2_excess'] for r in long),
        long_max_distance_excess=max(r['distance_excess'] for r in long),
        long_corrections_min=min(r['corrections'] for r in long),
        long_corrections_max=max(r['corrections'] for r in long),
        long_trial_limited=sum(r['stop_reason']=='trial_limit' for r in long),
        seed_check_samples=len(repeated),seed_check_seeds=[43,44],
        seed_check_max_distance2_excess=max(r['distance2_excess'] for r in repeated),
        metric_recheck_max_error=max(r['metric_recheck_error'] for r in rows+repeated+long),
        all_states_min_eig=min(r['min_eig'] for r in rows+repeated+long),
        all_states_min_pt_eig=min(r['min_pt_eig'] for r in rows+repeated+long),
        all_states_max_trace_error=max(r['trace_error'] for r in rows+repeated+long),
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        cargo=subprocess.check_output(['cargo','--version'],text=True).strip(),
        source_sha256={str(p.relative_to(BASELINE_RUST)):filehash(p)
                       for p in sorted(BASELINE_RUST.glob('src/*.rs'))},
        total_runs=len(rows+repeated+long))
    (HERE/'gilbert-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
