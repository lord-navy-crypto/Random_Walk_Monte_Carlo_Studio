from __future__ import annotations
import math, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import qmc
from .random_walk import simulate_endpoints, summarize_endpoints
from .monte_carlo import theoretical_nd_ball_volume
from .statistics import mean_ci_t


def bootstrap_mean_ci(values, confidence=0.95, n_boot=2000, seed=None):
    values=np.asarray(values,float)
    if values.size<2 or n_boot<100: raise ValueError("Need >=2 values and >=100 bootstrap resamples")
    rng=np.random.default_rng(seed); means=np.empty(n_boot)
    for i in range(n_boot): means[i]=rng.choice(values,size=values.size,replace=True).mean()
    alpha=1-confidence
    return {"mean":float(values.mean()),"confidence":confidence,"ci_low":float(np.quantile(means,alpha/2)),
            "ci_high":float(np.quantile(means,1-alpha/2)),"bootstrap_se":float(means.std(ddof=1))}


def multi_seed_random_walk(dim,n_steps,n_walkers,seeds,model="fixed",p1=1.0,p2=None,progress=None):
    seeds=list(seeds); rows=[]
    for i,seed in enumerate(seeds):
        t0=time.perf_counter(); ep,meta=simulate_endpoints(dim,n_steps,n_walkers,model,p1,p2,int(seed),return_metadata=True)
        s=summarize_endpoints(ep); s.update({"seed":int(seed),"runtime_seconds":time.perf_counter()-t0,"engine":meta['engine']}); rows.append(s)
        if progress: progress((i+1)/len(seeds),f"Independent seed {i+1}/{len(seeds)}")
    df=pd.DataFrame(rows); summary={}
    for col in ["mean_radius","std_radius","msd","runtime_seconds"]:
        summary[col+"_mean"]=float(df[col].mean()); summary[col+"_sd_across_seeds"]=float(df[col].std(ddof=1)) if len(df)>1 else 0.0
    return df,summary


def first_passage_1d(target,n_trials=5000,max_steps=100000,seed=None):
    """Extension module: first passage of a 1D fixed ±1 walk to either ±target."""
    target=int(target)
    if target<1 or n_trials<1 or max_steps<1: raise ValueError("Invalid inputs")
    rng=np.random.default_rng(seed); x=np.zeros(n_trials,np.int64); active=np.ones(n_trials,bool)
    hit_time=np.full(n_trials,-1,np.int64); hit_side=np.zeros(n_trials,np.int8)
    for step in range(1,int(max_steps)+1):
        idx=np.flatnonzero(active)
        if idx.size==0: break
        x[idx]+=np.where(rng.random(idx.size)<0.5,-1,1)
        hit=np.abs(x[idx])>=target
        if np.any(hit):
            hidx=idx[hit]; hit_time[hidx]=step; hit_side[hidx]=np.sign(x[hidx]).astype(np.int8); active[hidx]=False
    observed=hit_time>=0
    return {"target":target,"n_trials":int(n_trials),"max_steps":int(max_steps),"hit_fraction":float(observed.mean()),
            "censored_fraction":float((~observed).mean()),"mean_first_passage_among_hits":float(hit_time[observed].mean()) if observed.any() else None,
            "median_first_passage_among_hits":float(np.median(hit_time[observed])) if observed.any() else None,
            "right_hit_fraction_among_hits":float(np.mean(hit_side[observed]>0)) if observed.any() else None,"hit_times":hit_time[observed]}


def qmc_nd_ball_volume(dim,power=14,scramble=True,seed=None,max_coordinate_values=4_000_000):
    """Memory-bounded Sobol estimate using the first 2**power points of one scrambled sequence."""
    dim=int(dim); n=int(2**power)
    sampler=qmc.Sobol(d=dim,scramble=bool(scramble),seed=seed)
    chunk=max(1,int(max_coordinate_values)//dim); hits=0; done=0
    # scipy's random(n) advances the same Sobol sequence; concatenated chunks are the first N points.
    while done<n:
        k=min(chunk,n-done)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            u=sampler.random(n=k)
        pts=2*u-1
        hits += int(np.count_nonzero(np.einsum('ij,ij->i',pts,pts)<=1)); done += k
    est=(2.0**dim)*hits/n; true=theoretical_nd_ball_volume(dim)
    return {"dimension":dim,"n_samples":n,"effective_chunk_size":int(chunk),"hits_inside":hits,"estimate":float(est),"true_value":float(true),
            "absolute_error":float(abs(est-true)),"relative_error_percent":float(abs(est-true)/true*100) if true else float('nan'),"method":"Sobol QMC single scramble"}


def repeated_scrambled_qmc(dim,power=14,n_replicates=8,seed=None,confidence=0.95):
    """Independent Owen-scrambled Sobol replicates provide an empirical QMC uncertainty estimate."""
    if n_replicates<2: raise ValueError("Need >=2 scrambled replicates")
    rng=np.random.default_rng(seed); est=np.empty(n_replicates)
    for i in range(n_replicates):
        est[i]=qmc_nd_ball_volume(dim,power,True,int(rng.integers(0,2**32-1)))['estimate']
    mean=float(est.mean()); sd=float(est.std(ddof=1)); low,high=mean_ci_t(mean,sd,n_replicates,confidence); true=theoretical_nd_ball_volume(dim)
    return {"dimension":int(dim),"power":int(power),"samples_per_replicate":int(2**power),"n_replicates":int(n_replicates),
            "mean_estimate":mean,"replicate_sd":sd,"sem":sd/math.sqrt(n_replicates),"ci_low":low,"ci_high":high,
            "true_value":true,"absolute_error_of_mean":abs(mean-true),"estimates":est}
