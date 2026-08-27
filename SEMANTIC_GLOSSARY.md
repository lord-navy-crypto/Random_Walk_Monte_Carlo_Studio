# Semantic Glossary

V4 deliberately separates quantities that the fragments occasionally reuse or conflate.

| Symbol / term | Precise meaning | Not the same as |
|---|---|---|
| `d` | spatial dimension | sample count |
| `N_steps` | steps per random-walk trajectory | Monte Carlo samples |
| `M_walkers` | independent walkers in one ensemble | repeated full runs |
| `N_samples` | random/QMC points in one integration trial | random-walk steps |
| `R_trials` | independent repeated Monte Carlo trials | endpoint radius `R` |
| `H` | finite return-to-origin horizon | infinite-time recurrence |
| `K` | grid subdivisions per axis | total grid points (`K^d`) |
| `S` | non-negative step length before independent ± sign | signed displacement |
| radial SD | spread of walker endpoint radii | SEM of their mean |
| SEM | uncertainty scale of an estimated mean | population/walker spread |
| seed-to-seed SD | variation among full independent simulation runs | within-run radial SD |
| exact MSD theory | `E[R²]=N_steps E[S²]` | asymptotic mean-radius theory |
| mean-radius theory | CLT/chi large-N approximation | exact finite-N identity |
| return-by-H probability | returned at least once before finite horizon `H` | Pólya infinite-time recurrence probability |
| unit disk | `x²+y²≤1`; area `π` | boundary circle `x²+y²=1` |
| unit d-ball | `Σxᵢ²≤1`; d-volume `π^(d/2)/Γ(d/2+1)` | boundary sphere surface measure |

## Status labels

- **Source-derived**: directly reconstructs an experiment present in the supplied fragments.
- **Corrected interpretation**: same experiment, but ambiguous/mistaken mathematical terminology is made explicit.
- **Platform extension**: useful functionality not explicitly required by the fragments (multi-seed audit, bootstrap, first passage, repeated scrambled Sobol QMC, etc.).
