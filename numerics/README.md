# Minimum Hilbert--Schmidt distance: reproducibility code

Code and data for Palash Pandya's manuscript, *Minimum Hilbert--Schmidt
distance for pure states of Schmidt rank two*.

Manuscript code and data: <https://github.com/palashpandya/min-HSD-SR-2>.

Original Rust Gilbert implementation: <https://github.com/palashpandya/rusty-code>,
commit `8df88703d8c5e5cfca2e5b2cb34a2ef962ed3a3f`.

## Reproduce the calculations

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 numerics/rank2_validation.py
python3 numerics/gilbert_comparison.py
python3 numerics/gilbert_refinement.py --workers 8
python3 numerics/plot_gilbert_comparison.py
```

Rust/Cargo are also required. Cargo downloads the locked dependencies on its
first build. Both Gilbert scripts accept `--binary /path/to/rusty-code` to use
an existing executable. Each run has its own directory and config, so the
refinement script can safely run eight independent subprocesses concurrently.

`rank2_validation.py` performs unrestricted PPT optimization and symbolic,
formula, derivative and explicit separable-decomposition checks. It writes CSV
figure data, a JSON residual report and the analytical/PPT figures. The two
Gilbert scripts save the input matrices, final separable states, complete
trajectories, logs, configurations and independent matrix checks. The last
command regenerates the figures containing all three methods. Rerun it after
`rank2_validation.py`, which by itself restores the PPT-only comparison figure.

Python versions are pinned in `requirements.txt` and recorded in
`numerics/rank2-validation.json`. Rust versions, exact settings and actual
termination counts are recorded in the Gilbert JSON summaries. Timings depend
on the machine. Large refinement trajectories and logs are losslessly gzipped;
NumPy can read the `.csv.gz` trajectories directly.

## Initial comparison

`gilbert_comparison.py` builds `rusty-code-baseline/`, the exact source snapshot
used for the initial 102 runs. It uses the upstream `grad` optimizer with 20
alternating sweeps, three gradient steps per site per sweep, step size 0.3,
and raw random-product preselection. The source includes the documented loop
fix so every attempt advances the trial counter, plus an immediate exit at an
initially exact product input.

The baseline contains the same 75 alpha values as the PPT calculation, seed
42, at most 200 accepted corrections and 10,000,000 random trials per input.
It also contains 18 checks at seeds 43 and 44, and nine representative runs
with a 2,000-correction budget. Actual corrections can be fewer than the budget;
see `gilbert-comparison.csv` and `gilbert-summary.json`. The 1e-15 threshold is
on the squared distance itself, not a certified optimization-gap tolerance.

## Refined comparison

`gilbert_refinement.py` builds `rusty-code/`. It retains the SAME raw
preselection criterion:

`Tr[(rho - sigma)(tau - sigma)] > 0`.

Here sigma is the current iterate and tau is the sampled product projector.
Only passing trials are optimized. The refined implementation caches the
residual and evaluates the product-vector expectation directly, avoiding
unnecessary full-matrix construction for rejected trials. A Rust regression
test compares this expression with the original matrix calculation on 400
random cases in local dimensions two and three. Run it with:

```sh
cargo test --release --locked --manifest-path numerics/rusty-code/Cargo.toml
```

Defaults for the reported refinement are eight workers, 2,000 alternating
local sweeps, three gradient steps per site per sweep, step size 0.3,
100,000 accepted corrections, and 100,000,000 raw trials per input. All 75
PPT-grid inputs use seed 42; the Bell state additionally uses seeds 43 and 44.
The larger alternating-sweep budget is important: adding random trials alone
does not compensate for an insufficiently optimized product state.

An additional benchmark stopping rule ends a run when its squared distance
is within 1e-8 of the analytical value. That scalar reference is used ONLY
for stopping, never to initialize an iterate, propose a product state, or
choose convex weights. It is not an independently computed dual certificate.
Set `--gap 0` to disable it and run to the finite correction/trial budgets.
All limits can be changed through `--corrections`, `--trials`, `--sweeps`,
`--gradient-steps` and `--workers`. The final CSV and JSON summary record which
stopping condition actually occurred; reaching a budget is not convergence.

## Precision, validity and provenance

All calculations use double precision: Rust `f64` and `Complex64` (two f64
components), NumPy float64 and complex128. The historical Rust function
`hs_dist` and CSV column `distance` contain SQUARED Hilbert--Schmidt distance.
Every saved matrix is independently checked against that logged value using
the squared Frobenius norm. Checks also cover trace, Hermiticity, positivity,
PPT, monotone trajectories and the analytical lower bound.

Every iterate is separable by construction: the initial diagonal state is a
product-basis mixture and each update is a convex combination with a pure
product projector. PPT tests are additional checks, not the reason the
iterates are separable. The Gilbert outputs are upper bounds; the manuscript
proves the optimum separately.

Both Rust snapshots include an upstream commit, exact file hashes and a patch
in their provenance files. The baseline snapshot is preserved separately to
keep its original run records reproducible. The upstream `rusty-code`
repository is unchanged. Diagnostic filter-disabled experiments were used to
investigate performance; they are not part of the reported comparison.
