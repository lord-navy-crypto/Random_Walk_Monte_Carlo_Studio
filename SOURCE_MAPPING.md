# Source Mapping

This file separates what came from the supplied fragments from what the audited studio corrects or extends.

| Topic | Supplied source | Audited studio treatment |
|---|---|---|
| 2D fixed-step walk, N=10/100/1000/10000 | PDF + `f.py` | Preserved as default assignment scan |
| Mean radius, radial spread, MSD | PDF + `f.py` | Preserved; semantics separated |
| Uniform random step U(0.5,1.5) | PDF + `f.py` | Preserved; second-moment interpretation corrected |
| Dimensions 1–4 | PDF + `f.py` | Preserved; platform can extend higher |
| Return-to-origin simulation | PDF + `f.py` | Preserved as finite-horizon estimator |
| Circle area MC | PDF + `n.py` | Preserved |
| d-ball volume MC | PDF + `n.py` | Preserved; rare-event diagnostics added |
| 2D / 3D midpoint grid | PDF + `n.py` | Preserved; 3D memory use fixed |
| Error ~ N^-1/2 study | PDF + `n.py` | Preserved and upgraded to repeated-trial RMSE fitting |
| Error-bar plotting | notebook fragment | Reused for uncertainty-aware plots |
| 3D plotting | notebook fragment | Recognized as visualization intent; not misread as a new MC algorithm |
| Multi-seed, bootstrap, first passage, Sobol QMC | not explicit in source | Clearly labeled platform extensions |

## Documented source/reproduction discrepancies
- `f.py` hard-coded mean-radius coefficients are not consistently matched to its own axis-selection walk rule.
- The report's N=1000 fixed-step mean radius of 17.7 is not reproduced by the supplied walk rule (about 28.0).
- The report's d=2 return probability 0.988 at a 20,000-step horizon is not reproduced; larger ensembles give about 0.75–0.77.
- The report text says mean radius decreases with dimension, while direct simulation of its rule shows the opposite trend (approaching √N).


## Terminology normalization
- Source “circle area” → precise V4 label: **unit-disk area**, because the sampled set is `x²+y²≤1`.
- Source “N-dimensional sphere volume” → precise V4 label: **unit d-ball volume**, because the formula is `π^(d/2)/Γ(d/2+1)` for the enclosed d-dimensional volume.
- Legacy function names remain available so the supplied code lineage is still recognizable.
