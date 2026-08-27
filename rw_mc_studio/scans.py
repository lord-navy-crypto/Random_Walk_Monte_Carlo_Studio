from __future__ import annotations
import numpy as np
import pandas as pd
from .random_walk import simulate_endpoints, summarize_endpoints, theoretical_msd, theoretical_mean_radius
from .monte_carlo import repeated_circle_trials


def _apply_scan_value(cfg, variable, value):
    if variable == "n_steps": cfg["n_steps"] = int(value)
    elif variable == "dimension": cfg["dim"] = int(value)
    elif variable == "n_walkers": cfg["n_walkers"] = int(value)
    elif variable == "fixed_step_size":
        if cfg["model"] != "fixed": raise ValueError("fixed_step_size scan requires fixed model")
        cfg["p1"] = float(value)
    elif variable == "uniform_lower":
        if cfg["model"] != "uniform": raise ValueError("uniform_lower scan requires uniform model")
        cfg["p1"] = float(value)
    elif variable == "uniform_upper":
        if cfg["model"] != "uniform": raise ValueError("uniform_upper scan requires uniform model")
        cfg["p2"] = float(value)
    else: raise ValueError(f"Unsupported scan variable: {variable}")


def random_walk_scan(values, scan_variable="n_steps", *, dim=2, n_steps=1000, n_walkers=10000,
                     model="fixed", p1=1.0, p2=None, seed=12345, progress=None):
    values=list(values)
    if len(values)<1: raise ValueError("No scan values")
    rows=[]; master=np.random.default_rng(seed)
    for i,v in enumerate(values):
        cfg=dict(dim=dim,n_steps=n_steps,n_walkers=n_walkers,model=model,p1=p1,p2=p2)
        _apply_scan_value(cfg,scan_variable,v)
        ep,meta=simulate_endpoints(**cfg,seed=int(master.integers(0,2**32-1)),return_metadata=True)
        s=summarize_endpoints(ep)
        s.update({scan_variable:v,"theory_mean_radius_asymptotic":theoretical_mean_radius(cfg['dim'],cfg['n_steps'],cfg['model'],cfg['p1'],cfg['p2']),
                  "theory_msd_exact":theoretical_msd(cfg['n_steps'],cfg['model'],cfg['p1'],cfg['p2']),"engine":meta['engine']})
        rows.append(s)
        if progress: progress((i+1)/len(values),f"Scan point {i+1}/{len(values)}")
    return pd.DataFrame(rows)


def fit_power_law(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    mask=(x>0)&(y>0)&np.isfinite(x)&np.isfinite(y)
    if mask.sum()<3: raise ValueError("Power-law fit needs at least 3 positive finite points")
    lx,ly=np.log(x[mask]),np.log(y[mask]); alpha,log_a=np.polyfit(lx,ly,1)
    pred=log_a+alpha*lx; ss_res=float(np.sum((ly-pred)**2)); ss_tot=float(np.sum((ly-ly.mean())**2))
    return {"exponent":float(alpha),"prefactor":float(np.exp(log_a)),"r2_log":1-ss_res/ss_tot if ss_tot else 1.0,"n_fit":int(mask.sum())}


def mc_convergence_scan(values,n_trials=50,seed=12345,progress=None):
    values=list(values); rows=[]; rng=np.random.default_rng(seed)
    for i,n in enumerate(values):
        out=repeated_circle_trials(int(n),int(n_trials),int(rng.integers(0,2**32-1)))
        rows.append({k:out[k] for k in ["n_samples","mean_estimate","run_to_run_sd","sem_of_trial_mean","mean_estimate_ci_low","mean_estimate_ci_high","bias","mean_absolute_error","rmse"]})
        if progress: progress((i+1)/len(values),f"Convergence point {i+1}/{len(values)}")
    return pd.DataFrame(rows)
