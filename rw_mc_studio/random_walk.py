from __future__ import annotations
import math
from typing import Callable
import numpy as np
from scipy.special import gammaln
from .statistics import mean_ci_t, clopper_pearson

ProgressCallback = Callable[[float, str], None]


def _canonical_model(model: str) -> str:
    m = model.lower().strip()
    if m == "normal":  # backward compatibility with V1/V2
        return "folded_normal"
    return m


def step_second_moment(model: str, p1: float, p2: float | None = None) -> float:
    """Return E[S²] for the non-negative step-length distribution S."""
    model = _canonical_model(model)
    if model == "fixed":
        if p1 < 0: raise ValueError("Fixed step length must be >= 0")
        return float(p1) ** 2
    if model == "uniform":
        if p2 is None or p1 < 0 or p2 <= p1:
            raise ValueError("Uniform step length requires 0 <= a < b")
        a, b = float(p1), float(p2)
        return (a*a + a*b + b*b) / 3.0
    if model == "folded_normal":
        if p2 is None or p2 < 0:
            raise ValueError("Folded normal requires sigma >= 0")
        # |X|² = X², so E[S²] = μ² + σ² exactly.
        return float(p1) ** 2 + float(p2) ** 2
    if model == "exponential":
        if p1 <= 0: raise ValueError("Exponential scale must be > 0")
        return 2.0 * float(p1) ** 2
    raise ValueError(f"Unsupported step model: {model}")


def theoretical_msd(n_steps: int, model: str = "fixed", p1: float = 1.0, p2: float | None = None) -> float:
    """Exact ensemble MSD for the symmetric axis-aligned walk: E[R²]=N E[S²]."""
    if n_steps < 0: raise ValueError("n_steps must be >= 0")
    return float(n_steps) * step_second_moment(model, p1, p2)


def theoretical_mean_radius(dim: int, n_steps: int, model: str = "fixed",
                            p1: float = 1.0, p2: float | None = None) -> float:
    """CLT/chi asymptotic mean endpoint radius, not an exact finite-N identity."""
    if dim < 1 or n_steps < 0:
        raise ValueError("dim must be >=1 and n_steps >=0")
    if n_steps == 0: return 0.0
    e_s2 = step_second_moment(model, p1, p2)
    sigma_coord = math.sqrt(n_steps * e_s2 / dim)
    log_chi_mean = 0.5*math.log(2.0) + gammaln((dim+1)/2.0) - gammaln(dim/2.0)
    return float(sigma_coord * math.exp(log_chi_mean))


def _sample_sizes(rng: np.random.Generator, shape, model: str, p1: float, p2: float | None):
    model = _canonical_model(model)
    if model == "fixed":
        return np.full(shape, float(p1), dtype=np.float64)
    if model == "uniform":
        if p2 is None or p1 < 0 or p2 <= p1: raise ValueError("Require 0 <= a < b")
        return rng.uniform(float(p1), float(p2), size=shape)
    if model == "folded_normal":
        if p2 is None or p2 < 0: raise ValueError("Require sigma >= 0")
        return np.abs(rng.normal(float(p1), float(p2), size=shape))
    if model == "exponential":
        if p1 <= 0: raise ValueError("Scale must be > 0")
        return rng.exponential(float(p1), size=shape)
    raise ValueError(f"Unsupported step model: {model}")


def _fixed_endpoints_multinomial(rng, dim: int, n_steps: int, n_walkers: int, step_size: float, max_category_values: int = 4_000_000):
    """Exact endpoint-only accelerator, blocked to cap temporary multinomial count memory."""
    probs = np.full(2*dim, 1.0/(2*dim), dtype=float)
    coords = np.empty((n_walkers, dim), dtype=np.float64)
    block = max(1, min(n_walkers, int(max_category_values)//(2*dim)))
    for w0 in range(0, n_walkers, block):
        w1 = min(w0+block, n_walkers)
        counts = rng.multinomial(int(n_steps), probs, size=int(w1-w0))
        coords[w0:w1] = (counts[:,0::2]-counts[:,1::2]).astype(np.float64)*float(step_size)
    return coords


def simulate_endpoints(dim: int, n_steps: int, n_walkers: int = 10_000,
                       model: str = "fixed", p1: float = 1.0, p2: float | None = None,
                       seed: int | None = None, max_block_draws: int = 2_000_000,
                       max_endpoint_values: int = 10_000_000,
                       progress: ProgressCallback | None = None,
                       return_metadata: bool = False):
    """
    Simulate independent endpoint positions.

    Engine semantics:
    - fixed steps: exact multinomial endpoint accelerator (same endpoint distribution as stepwise simulation)
    - random step lengths: memory-bounded direct simulation in walker/step blocks
    """
    if dim < 1 or n_steps < 0 or n_walkers < 1:
        raise ValueError("Require dim>=1, n_steps>=0, n_walkers>=1")
    if int(dim)*int(n_walkers) > int(max_endpoint_values):
        mb = int(dim)*int(n_walkers)*8/1_000_000
        raise ValueError(f"Endpoint array would require about {mb:.1f} MB; reduce walkers or dimension.")
    model = _canonical_model(model)
    _ = step_second_moment(model, p1, p2)  # validation
    rng = np.random.default_rng(seed)
    if n_steps == 0:
        coords = np.zeros((n_walkers, dim), dtype=float)
        meta = {"engine": "zero-step", "exact_endpoint_engine": True}
        return (coords, meta) if return_metadata else coords

    if model == "fixed":
        coords = _fixed_endpoints_multinomial(rng, dim, n_steps, n_walkers, p1)
        if progress: progress(1.0, "Exact fixed-step endpoint sampling complete")
        meta = {"engine": "multinomial-endpoint", "exact_endpoint_engine": True}
        return (coords, meta) if return_metadata else coords

    coords = np.zeros((n_walkers, dim), dtype=np.float64)
    # Bound temporary arrays by max_block_draws. We block walkers first, then steps.
    walker_block = max(1, min(n_walkers, int(math.sqrt(max_block_draws))))
    total_work = n_walkers * n_steps
    completed = 0
    for w0 in range(0, n_walkers, walker_block):
        w1 = min(w0 + walker_block, n_walkers)
        wb = w1 - w0
        step_block = max(1, min(n_steps, max_block_draws // wb))
        block_coords = np.zeros((wb, dim), dtype=np.float64)
        for s0 in range(0, n_steps, step_block):
            k = min(step_block, n_steps-s0)
            axes = rng.integers(0, dim, size=(wb, k), dtype=np.int32)
            signs = np.where(rng.random((wb, k)) < 0.5, -1.0, 1.0)
            sizes = _sample_sizes(rng, (wb, k), model, p1, p2)
            increments = signs * sizes
            linear = (np.arange(wb, dtype=np.int64)[:, None] * dim + axes).ravel()
            acc = np.bincount(linear, weights=increments.ravel(), minlength=wb*dim).reshape(wb, dim)
            block_coords += acc
            completed += wb*k
            if progress: progress(min(completed/total_work, 1.0), "Simulating random-step ensemble")
        coords[w0:w1] = block_coords
    meta = {"engine": "blocked-direct", "exact_endpoint_engine": True,
            "max_block_draws": int(max_block_draws)}
    return (coords, meta) if return_metadata else coords


def summarize_endpoints(endpoints: np.ndarray, confidence: float = 0.95) -> dict:
    r = np.linalg.norm(np.asarray(endpoints, dtype=float), axis=1)
    n = int(len(r))
    if n < 1: raise ValueError("No endpoints")
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sem = std/math.sqrt(n) if n > 1 else 0.0
    low, high = mean_ci_t(mean, std, n, confidence)
    return {
        "n_walkers": n,
        "mean_radius": mean,
        "median_radius": float(np.median(r)),
        "std_radius": std,
        "variance_radius": std**2,
        "rms_radius": float(np.sqrt(np.mean(r*r))),
        "msd": float(np.mean(r*r)),
        "sem_radius": sem,
        "confidence_level": confidence,
        "mean_radius_ci_low": low,
        "mean_radius_ci_high": high,
    }


def simulate_trajectory(dim: int, n_steps: int, model: str = "fixed",
                        p1: float = 1.0, p2: float | None = None,
                        seed: int | None = None) -> np.ndarray:
    if dim < 1 or n_steps < 0: raise ValueError("Invalid dimension or step count")
    rng = np.random.default_rng(seed)
    coords = np.zeros((n_steps+1, dim), dtype=float)
    for i in range(1, n_steps+1):
        axis = int(rng.integers(0, dim))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        size = float(_sample_sizes(rng, 1, model, p1, p2)[0])
        coords[i] = coords[i-1]
        coords[i, axis] += sign*size
    return coords


def return_probability(dim: int, max_steps: int = 5000, n_trials: int = 1000,
                       seed: int | None = None, confidence: float = 0.95,
                       progress: ProgressCallback | None = None) -> dict:
    """Finite-horizon P(return to origin at least once by step H), not infinite-time Pólya probability."""
    if dim < 1 or max_steps < 1 or n_trials < 1: raise ValueError("Invalid inputs")
    rng = np.random.default_rng(seed)
    coords = np.zeros((n_trials, dim), dtype=np.int64)
    active = np.ones(n_trials, dtype=bool)
    returned = np.zeros(n_trials, dtype=bool)
    first_return = np.full(n_trials, -1, dtype=np.int64)
    stride = max(1, max_steps//100)
    for step in range(1, max_steps+1):
        idx = np.flatnonzero(active)
        if idx.size == 0: break
        axes = rng.integers(0, dim, size=idx.size)
        signs = np.where(rng.random(idx.size) < 0.5, -1, 1)
        coords[idx, axes] += signs
        hit = np.all(coords[idx] == 0, axis=1)
        if np.any(hit):
            hidx = idx[hit]
            returned[hidx] = True
            first_return[hidx] = step
            active[hidx] = False
        if progress and (step % stride == 0 or step == max_steps):
            progress(step/max_steps, "Estimating finite-horizon return probability")
    k = int(returned.sum())
    p = k/n_trials
    low, high = clopper_pearson(k, n_trials, confidence)
    return {
        "dimension": int(dim), "horizon_steps": int(max_steps), "n_trials": int(n_trials),
        "returned_trials": k, "return_probability_by_horizon": float(p),
        "confidence_level": confidence, "probability_ci_low": low, "probability_ci_high": high,
        "mean_first_return_step_among_returns": float(first_return[returned].mean()) if returned.any() else None,
        "median_first_return_step_among_returns": float(np.median(first_return[returned])) if returned.any() else None,
        "semantic_note": "Finite-horizon estimate; not the exact infinite-time Pólya recurrence probability.",
    }
