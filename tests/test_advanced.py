import math
import numpy as np
from rw_mc_studio.advanced import bootstrap_mean_ci,first_passage_1d,repeated_scrambled_qmc
from rw_mc_studio.scans import fit_power_law,random_walk_scan


def test_bootstrap_ci():
    x=np.arange(1,101,dtype=float); out=bootstrap_mean_ci(x,n_boot=500,seed=1); assert out['ci_low']<x.mean()<out['ci_high']

def test_first_passage_target_one():
    out=first_passage_1d(1,n_trials=100,max_steps=10,seed=3); assert out['hit_fraction']==1 and out['mean_first_passage_among_hits']==1

def test_repeated_qmc_circle_reasonable():
    out=repeated_scrambled_qmc(2,power=10,n_replicates=4,seed=2); assert abs(out['mean_estimate']-math.pi)<0.08

def test_power_law_fit():
    x=np.array([1,4,16,64],float); y=3*x**0.5; out=fit_power_law(x,y); assert abs(out['exponent']-0.5)<1e-10

def test_uniform_scan_semantic_validation():
    df=random_walk_scan([0.2,0.4],"uniform_lower",dim=2,n_steps=20,n_walkers=1000,model='uniform',p1=0.1,p2=1.5,seed=1)
    assert len(df)==2 and 'uniform_lower' in df
