# Validation Results

Final reproducible checks for the V4 visual-first English edition. The V3 numerical and memory-safety audit remains the computational baseline.

| Check | Final result | Interpretation |
|---|---:|---|
| Fixed-step 2D scan N=10..10000, fitted alpha | 0.5007 | agrees with sqrt(N) diffusion scaling |
| Uniform U(0.5,1.5), N=1000 MSD | 1084.97 | exact target 1083.33; ~0.15% relative deviation |
| 2D midpoint grid K=100 | 3.144000 | exactly reproduces supplied result |
| 3D midpoint grid K=50 | 4.208128 | exactly reproduces supplied result |
| Repeated-MC RMSE exponent | -0.5024 | agrees with N^-1/2 Monte Carlo scaling |
| d=2, H=20000, 5000 trajectories return-by-horizon | 0.7632 | 95% exact-binomial CI [0.7512,0.7749]; does not reproduce reported 0.988 |
| d=20 rejection MC, N=10000 | 0 observed hits | exact-binomial upper volume bound remains positive; expected hits only ~0.000246 |

## Fixed-step source-like scan

With 20,000 walkers and the supplied axis-selection rule:

| N_steps | Simulated mean R | Simulated MSD | Asymptotic mean R | Exact MSD |
|---:|---:|---:|---:|---:|
| 10 | 2.7884 | 9.9520 | 2.8025 | 10 |
| 100 | 8.8232 | 98.9794 | 8.8623 | 100 |
| 1000 | 27.8752 | 991.3313 | 28.0250 | 1000 |
| 10000 | 88.6941 | 10020.9615 | 88.6227 | 10000 |

The power-law fit gives `alpha = 0.5007`.

## Return-probability source-parameter reproduction

Using source-like horizons/trial counts with fixed deterministic seeds:

| d | Horizon H | Trials | Reproduced P(return by H) |
|---:|---:|---:|---:|
| 1 | 20000 | 400 | 0.9925 |
| 2 | 20000 | 400 | 0.7725 |
| 3 | 5000 | 200 | 0.3600 |
| 4 | 5000 | 200 | 0.1750 |

The supplied report gives approximately 0.995, 0.988, 0.340, 0.195. d=3 and d=4 are consistent with finite Monte Carlo noise at these small trial counts; d=2 is not. A larger d=2 run with 5,000 trajectories gives 0.7632 with 95% exact-binomial CI [0.7512, 0.7749].

## Geometry terminology check

The source's `x²+y²≤1` estimator is an area estimator for the **unit disk**, even though the legacy wording says “circle area.” The d-dimensional formula `π^(d/2)/Γ(d/2+1)` is the enclosed **unit d-ball volume**, not the surface measure of the boundary sphere. V4 preserves legacy backend names for traceability but uses precise user-facing terminology.

## Verification gates

- `python -m compileall -q rw_mc_studio app.py`: **PASS**
- Project test functions: **18/18 PASS** in the packaging environment
- `python -m pytest -q`: configured in CI and the local verification command
- Source-equivalent 2D/3D grid outputs: **PASS**
- Fixed/random-step MSD checks: **PASS**
- MC convergence exponent check: **PASS**
- Exact-binomial zero-hit interval check: **PASS**
- Semantic alias/confidence validation tests: **PASS**
- English-only repository scan outside the supplied PDF: **PASS**
- Notebook JSON, preset JSON, and `pyproject.toml` parse: **PASS**
- macOS/Linux launcher shell syntax: **PASS**
- Streamlit live-start: **not executable in this isolated runner** because Streamlit is not preinstalled. GitHub CI installs requirements and performs an `import app` smoke check.
