from __future__ import annotations
import math
import numpy as np
from scipy.special import gammaln
from .statistics import clopper_pearson, mean_ci_t


def log_theoretical_nd_ball_volume(dim: int) -> float:
    if dim < 1: raise ValueError("dim must be >=1")
    return float((dim/2.0)*math.log(math.pi) - gammaln(dim/2.0 + 1.0))


def theoretical_nd_ball_volume(dim: int) -> float:
    return float(math.exp(log_theoretical_nd_ball_volume(dim)))


def _mc_from_hits(hits: int, n_samples: int, cube_volume: float, true_value: float, confidence=0.95):
    p = hits/n_samples
    p_low, p_high = clopper_pearson(hits, n_samples, confidence)
    est = cube_volume*p
    empirical_se = cube_volume*math.sqrt(max(p*(1-p), 0.0)/n_samples)
    return {
        "estimate": float(est), "true_value": float(true_value),
        "absolute_error": float(abs(est-true_value)),
        "relative_error_percent": float(abs(est-true_value)/true_value*100) if true_value else float('nan'),
        "hits_inside": int(hits), "inside_fraction": float(p),
        "mc_standard_error_plugin": float(empirical_se),
        "confidence_level": confidence,
        "volume_ci_low_exact_binomial": float(cube_volume*p_low),
        "volume_ci_high_exact_binomial": float(cube_volume*p_high),
    }


def circle_area_mc(n_samples: int, seed: int | None = None, return_points: bool = False,
                   chunk_size: int = 500_000, confidence: float = 0.95):
    if n_samples < 1: raise ValueError("n_samples must be >=1")
    rng = np.random.default_rng(seed)
    if return_points:
        if n_samples > 500_000:
            raise ValueError("Point visualization is capped at 500,000 samples to protect memory.")
        pts = rng.uniform(-1.0, 1.0, size=(n_samples,2))
        inside = np.einsum('ij,ij->i', pts, pts) <= 1.0
        out = _mc_from_hits(int(inside.sum()), n_samples, 4.0, math.pi, confidence)
        out['points'], out['inside_mask'] = pts, inside
        out['semantic_note'] = "Estimator is area of the unit disk x²+y²≤1; the legacy source/function name says circle area."
        return out
    hits=0; done=0
    while done<n_samples:
        k=min(chunk_size,n_samples-done)
        pts=rng.uniform(-1.0,1.0,size=(k,2))
        hits += int(np.count_nonzero(np.einsum('ij,ij->i',pts,pts)<=1.0))
        done += k
    out = _mc_from_hits(hits,n_samples,4.0,math.pi,confidence)
    out['semantic_note'] = "Estimator is area of the unit disk x²+y²≤1; the legacy source/function name says circle area."
    return out


# Semantic alias; legacy circle_area_mc name is retained for compatibility.
disk_area_mc = circle_area_mc


def nd_ball_volume_mc(dim: int, n_samples: int, seed: int | None = None,
                      chunk_size: int = 200_000, confidence: float = 0.95,
                      max_coordinate_values: int = 4_000_000) -> dict:
    if dim < 1 or n_samples < 1: raise ValueError("dim,n_samples must be positive")
    rng=np.random.default_rng(seed); hits=0; done=0
    safe_chunk=max(1,min(int(chunk_size), int(max_coordinate_values)//int(dim)))
    while done<n_samples:
        k=min(safe_chunk,n_samples-done)
        pts=rng.uniform(-1.0,1.0,size=(k,dim))
        hits += int(np.count_nonzero(np.einsum('ij,ij->i',pts,pts)<=1.0))
        done += k
    cube=2.0**dim; true=theoretical_nd_ball_volume(dim)
    out=_mc_from_hits(hits,n_samples,cube,true,confidence)
    expected_hits=float(n_samples*true/cube)
    out.update({
        "dimension": int(dim), "n_samples": int(n_samples), "effective_chunk_size": int(safe_chunk), "expected_hits_if_model_exact": expected_hits,
        "rare_event_status": "LOW_EXPECTED_HITS" if expected_hits < 10 else "OK",
        "semantic_note": "Hypercube rejection sampling becomes a rare-event estimator in high dimension; zero hits do not imply zero uncertainty.",
    })
    return out


def repeated_circle_trials(n_samples: int, n_trials: int = 100, seed: int | None = None,
                           confidence: float = 0.95, fast_equivalent: bool = True) -> dict:
    """
    Repeated independent area estimates.
    fast_equivalent=True samples hit counts from Binomial(N, π/4), exactly matching the distribution
    of the source point-in-square estimator while avoiding repeated point allocation.
    """
    if n_samples < 1 or n_trials < 2: raise ValueError("Require n_samples>=1 and n_trials>=2")
    rng=np.random.default_rng(seed)
    if fast_equivalent:
        hits=rng.binomial(n_samples, math.pi/4.0, size=n_trials)
        estimates=4.0*hits/n_samples
        engine="binomial-distribution-equivalent"
    else:
        estimates=np.array([circle_area_mc(n_samples,int(rng.integers(0,2**32-1)))['estimate'] for _ in range(n_trials)])
        engine="geometric-point-sampling"
    mean=float(estimates.mean()); sd=float(estimates.std(ddof=1)); sem=sd/math.sqrt(n_trials)
    low,high=mean_ci_t(mean,sd,n_trials,confidence)
    err=estimates-math.pi
    return {
        "n_samples": int(n_samples), "n_trials": int(n_trials), "engine": engine,
        "mean_estimate": mean, "run_to_run_sd": sd, "sem_of_trial_mean": sem,
        "confidence_level": confidence, "mean_estimate_ci_low": low, "mean_estimate_ci_high": high,
        "true_value": math.pi, "bias": float(err.mean()),
        "mean_absolute_error": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err*err))),
        "estimates": estimates,
    }


def grid_circle_area(k: int) -> dict:
    """Memory-safe 2D midpoint grid using rows instead of KxK mesh arrays."""
    if k < 1: raise ValueError("k>=1")
    x=np.linspace(-1+1/k,1-1/k,k); x2=x*x; hits=0
    for y in x: hits += int(np.count_nonzero(x2+y*y<=1))
    est=hits*(2/k)**2
    return {"estimate":float(est),"true_value":math.pi,"absolute_error":float(abs(est-math.pi)),"points":int(k*k)}


# Semantic alias; legacy grid_circle_area name is retained for compatibility.
grid_disk_area = grid_circle_area


def grid_sphere_volume(k: int) -> dict:
    """Memory-safe 3D midpoint grid using z slices instead of allocating K³ coordinate arrays."""
    if k < 1: raise ValueError("k>=1")
    x=np.linspace(-1+1/k,1-1/k,k)
    xx,yy=np.meshgrid(x,x,indexing='xy'); r2xy=xx*xx+yy*yy
    hits=0
    for z in x:
        hits += int(np.count_nonzero(r2xy+z*z<=1))
    est=hits*(2/k)**3; true=4*math.pi/3
    return {"estimate":float(est),"true_value":true,"absolute_error":float(abs(est-true)),"points":int(k**3)}
