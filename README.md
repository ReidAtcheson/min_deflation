# min_deflation

This repository contains small utilities and experiments used to explore how
deflation vectors influence the convergence of the Conjugate Gradient (CG)
method. The code here is intentionally minimal and meant for rapid
experimentation, not production use.

## What is here?

* `util.py` provides helper functions
  * `chebyshev_diagonal_spd` constructs a diagonal SPD matrix whose
eigenvalues follow a Chebyshev distribution. It is useful for quickly
creating matrices with known conditioning.
  * `cg_residuals` runs SciPy's CG solver on such a matrix and records the
residual at each iteration.
* `experiments/`
  * `compare_residuals.py` plots residual curves for different choices of the
deflation parameter `k`.
  * `regression_experiment.py` samples random problem parameters and fits a
simple linear regression model predicting the number of CG iterations needed
to reach a residual threshold. The script also reports the resulting R^2
value.
* `tests/` contains a couple of smoke tests to ensure the helper functions and
experiments run as expected.

Again, everything here is for quick testing. The code is short, lacks robust
error handling, and should not be considered production ready.
