# V3 Full Audit Report

## Critical issues found and fixed

### 1. Ambiguous `sigma_s` meaning
The source text describes the standard deviation of `U(0.5,1.5)` (~0.2887), while the original Python computes `sqrt(E[S²])` but names it `sigma_s`. Those are different quantities. V3 uses explicit names: `step_second_moment`, `E[S²]`, and `radial spread SD`.

### 2. Exact vs asymptotic theory was mixed
`MSD = N E[S²]` is exact for the symmetric walk model. The chi-distribution formula for mean radius is a CLT/large-N approximation. V3 labels each result accordingly.

### 3. Pólya terminology was too strong
The supplied code stops at a finite `max_steps`; therefore it estimates `P(return by H)`, not the exact infinite-time recurrence probability. V3 renames output fields and labels the limitation.

### 4. Zero hits in high dimension
The plug-in binomial standard error is zero when the observed hit fraction is zero, which is misleading. V3 adds exact Clopper-Pearson intervals and expected-hit diagnostics.

### 5. 3D grid memory risk
The original `meshgrid(x,y,z)` creates multiple K³ arrays. V3 slices over z and only keeps K² arrays in memory.

### 6. Random-walk endpoint memory risk
The V2 chunking still allowed temporary arrays proportional to walkers × chunk_steps. V3 bounds temporary draws and blocks both walkers and steps.

### 7. Fixed-step simulation was unnecessarily O(MN)
For endpoint-only statistics, the counts of ± moves along each axis follow a multinomial distribution exactly. V3 uses that exact endpoint engine for fixed steps.

### 8. Uniform scan UI could generate invalid ranges
V2 exposed generic `step_size` while using `p1/p2`, making scans ambiguous and capable of crossing the fixed upper bound. V3 exposes `uniform_lower` and `uniform_upper` separately and validates `a<b`.

### 9. Statistical terms were conflated
V3 separates:
- walker radial SD,
- SEM of the run mean,
- t-based CI of the run mean,
- seed-to-seed SD,
- bootstrap CI across run means,
- QMC scramble-to-scramble SD.

### 10. QMC uncertainty
A single Sobol run does not justify a plain MC binomial CI. V3 adds repeated independently scrambled Sobol replicates and estimates uncertainty across those replicates.

## Verification gates
- Syntax compilation
- Unit/integration tests
- Streamlit startup smoke test
- Numerical smoke tests for source experiments

### 11. Reported 2D return probability is not reproducible from the stated finite-horizon method
The supplied report lists d=2 return probability as 0.988 while its method uses a 20,000-step horizon. Re-running the same model with a larger 5,000-trajectory ensemble gives about 0.763 (95% exact-binomial CI about 0.751–0.775). The V3 platform therefore does **not** hard-code 0.988 as a target; it reports the finite-horizon estimate and its CI. This is kept as a source-vs-reproduction discrepancy rather than silently rewriting the source.

## Local verification performed in this rebuild
- `python -m compileall`: PASS
- `python -m pytest -q`: 18/18 PASS after the final semantic/memory-safety patch
- Source-like fixed-step scan: fitted exponent α ≈ 0.5006
- U(0.5,1.5), N=1000: simulated MSD ≈ 1084.49 vs exact 1083.33 (~0.11% difference)
- Grid K=100: 3.144000 (matches supplied result)
- 3D grid K=50: 4.208128 (matches supplied result)
- Repeated-MC RMSE fit: β ≈ -0.5008
- Streamlit live-start could not be executed in the isolated local runner because Streamlit is not preinstalled and outbound package installation is unavailable; GitHub CI includes an app-import smoke check after installing requirements.

### 12. Mean-radius coefficients and reported N=1000 values are internally inconsistent
The supplied `f.py` uses coefficients 1.1284 (d=1), 0.7979 (d=2), 0.9213 (d=3), and 1.0 (d>=4). For the actual code rule—choose one of d axes uniformly and then move ±1—the CLT/chi asymptotic coefficients are approximately 0.7979, 0.8862, 0.9213, and 0.9400 for d=1..4. A 100,000-walker reproduction at N=1000 gave mean radii about 25.30, 28.04, 29.12, 29.81, respectively, agreeing with the corrected coefficients. The source report's statement that mean distance decreases with dimension, and its 17.7 fixed-step mean at N=1000, do not match the supplied simulation rule.

This is a source inconsistency, not a feature to preserve as a numerical target. V3 preserves the experiment but labels corrected theory separately.


### 13. “Circle” vs “disk”; “sphere” vs “ball” terminology
The supplied material says “circle area” while sampling `x²+y²≤1`, which is mathematically the **unit disk**. It also says “N-dimensional sphere volume” while using `V_d=π^(d/2)/Γ(d/2+1)`, which is the d-dimensional volume of the **unit d-ball**, not the surface measure of the boundary sphere. V3 keeps legacy function names for compatibility but uses precise user-facing labels and semantic aliases.

### 14. Symbol overload
The fragments reuse `N` for random-walk steps and Monte Carlo sample count. V3 distinguishes `N_steps`, `M_walkers`, `N_samples`, `R_trials`, `H` (return horizon), `K` (grid subdivisions), and `d` (dimension) in the semantic glossary and UI.
