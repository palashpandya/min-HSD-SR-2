# Minimum Hilbert--Schmidt distance: reproducibility code

Code and data for Palash Pandya's manuscript, *Minimum Hilbert--Schmidt
distance for pure states of Schmidt rank two*.

## Reproduce the calculations

From the root of the code archive (which contains `numerics/` and `gfx/`):

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 numerics/rank2_validation.py
python3 numerics/gilbert_comparison.py
```

The first script performs independent unrestricted PPT optimization, symbolic
checks, and formula/decomposition checks. It writes CSV data, a JSON residual
report, and the analytical/PPT figures. The second builds the Rust snapshot
using its Cargo lockfile, runs the Gilbert comparison, saves every raw output,
and writes `gilbert-comparison.csv` and `gilbert-summary.json`. Cargo requires
network access on its first build. An existing executable can be supplied as
`python3 numerics/gilbert_comparison.py --binary /path/to/rusty-code`.

Run `python3 numerics/plot_gilbert_comparison.py` last to regenerate the figures
containing all three methods. Rerunning the first script alone restores the
analytical/PPT-only version of the comparison figure; the last command adds
the Gilbert data again.

The environment used for the recorded Python results is listed in
`rank2-validation.json`. The Rust compiler and numerical protocol are recorded
in `gilbert-summary.json`. Elapsed times are machine-dependent.

## Gilbert implementation and exact version

Upstream repository: <https://github.com/palashpandya/rusty-code>.

Upstream commit: `8df88703d8c5e5cfca2e5b2cb34a2ef962ed3a3f`.

The `rusty-code/` directory is a source snapshot of this commit with the
small changes in `rusty-code/manuscript-loop-fix.patch`. These changes sample
a fresh product state and advance the trial counter on every attempt, stop
immediately for an initially exact product input, and report final counts.
The upstream repository is unchanged. File hashes and provenance are saved
in `rusty-code/provenance.json`.

The historical Rust function `hs_dist` and CSV column `distance` return
**squared** HSD. The comparison script independently recomputes the squared
Frobenius norm from every saved density matrix. It also checks normalization,
Hermiticity, positivity, PPT, trajectory monotonicity, and the analytical lower
bound. The iterates are separable by construction: the initial diagonal state
is a product-basis mixture, and each update is a convex combination with a
product projector. PPT checks are additional numerical checks, not the reason
the iterates are separable.

## Sampling and stopping protocol

- Baseline: the same 75 alpha values as the PPT calculation, seed 42, at most
  200 accepted corrections and 10,000,000 random trials per input.
- Seed checks: alpha = 0.01, 0.05, 0.2, 0.3, alpha_star, 0.4, 0.5, 0.6,
  and 1/sqrt(2), seeds 43 and 44, with the same 200-correction budget.
- Longer runs: the same nine representative inputs, seed 42, at most 2,000
  accepted corrections and 10,000,000 trials.
- The `grad` product optimizer uses 20 local sweeps, three gradient steps per
  site per sweep, and learning rate 0.3. The upstream random-product
  preselection filter is retained.
- The threshold 1e-15 applies to the squared distance itself. It is **not** a
  certified optimization-gap tolerance. The other stopping conditions are
  finite budgets. Check `corrections` and `stop_reason` for actual termination.

Each directory under `gilbert-runs/` includes the input matrix, relative-path
configuration, complete distance trajectory, final separable state, process
logs, and checked result. Randomness is fixed by the seed and the locked Rust
dependencies. The baseline and longer run at a given input and seed have the
same trajectory until the shorter run stops. These calculations provide
numerical upper bounds; the manuscript proves the optimum independently.

## Repository publication

This archive is prepared for the manuscript-specific GitHub repository that
the author will create. Its final URL and release identifier should be added
to the manuscript alongside the existing Rust implementation repository.
