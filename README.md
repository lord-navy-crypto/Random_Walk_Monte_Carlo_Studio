# Random Walk & Monte Carlo Simulation Studio

An English-only interactive workbench for random walks, Monte Carlo integration, uncertainty analysis, and reproducible numerical experiments.

## V4 result design

Every experiment follows the same research-friendly reading order:

1. **Visualization** — see the distribution, curve, trajectory, or comparison first.
2. **Key results** — read the small set of decision-relevant metrics next.
3. **Complete data** — inspect and export the full numerical table last.

Numbers use consistent precision, scientific notation for extreme magnitudes, thousands separators for counts, explicit confidence intervals, and precise geometry/statistics terminology. Every result table can be downloaded as CSV.

## Included experiments

- Random-walk trajectories and endpoint ensembles
- Fixed and uniformly distributed step lengths
- Parameter scans with theory comparisons
- Repeated unit-disk Monte Carlo and convergence-rate estimation
- High-dimensional unit-ball estimation with pseudorandom MC and scrambled Sobol QMC
- Finite-horizon return-to-origin and one-dimensional first-passage experiments
- Midpoint-grid versus Monte Carlo comparisons
- Multi-seed reproducibility and bootstrap uncertainty
- Validated JSON preset import and export

## Numerical safeguards

- Exact multinomial endpoint acceleration for fixed-step endpoint ensembles
- Memory-bounded blocks for random-length steps
- Student-t intervals for ensemble means
- Exact Clopper-Pearson intervals and rare-event diagnostics
- Independent scrambled Sobol replicates for QMC uncertainty
- Memory-safe three-dimensional grid slicing
- Explicit separation of radial SD, SEM, seed-to-seed SD, bootstrap CI, and QMC replicate uncertainty
- Precise distinction between unit-disk area and unit d-ball volume

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Convenience launchers are included: double-click `run_macos.command` on macOS, run `./run_linux.sh` on Linux, or double-click `run_windows.bat` on Windows. The first launch creates a local virtual environment and installs dependencies.

## Verify

```bash
python -m compileall -q rw_mc_studio app.py
python -m pytest -q
```

GitHub Actions runs compilation, tests, and an application-import smoke check.

## Repository map

```text
app.py                    Streamlit interface
rw_mc_studio/             Numerical and statistical library
tests/                    Automated tests
.github/workflows/ci.yml  Continuous integration
examples/                 Example preset and plotting fragment
source_fragments/         Supplied reconstruction references
AUDIT_REPORT.md           Numerical audit and corrections
SOURCE_MAPPING.md         Source-to-implementation mapping
SEMANTIC_GLOSSARY.md      Symbol and terminology definitions
VALIDATION_RESULTS.md     Reproducible verification record
LICENSE                   MIT license for original repository work
NOTICE.md                 Copyright and third-party material notice
```

## Reproducibility

Each stochastic module accepts a seed. Exported tables contain the parameters and statistics needed to compare runs. A matching seed and configuration reproduce the same pseudorandom result under a compatible dependency stack.

## License and source material

Original code and documentation created for this repository are available under the MIT License. Supplied reference fragments in `source_fragments/` remain subject to any rights held by their original authors and are included for traceability; the MIT License does not automatically relicense third-party material. See `NOTICE.md`.

## Citation

If this project supports a report or class project, cite the repository URL, release tag, version `4.0.0`, and access date. Also cite the original academic sources for any theory or supplied material used in the report.
