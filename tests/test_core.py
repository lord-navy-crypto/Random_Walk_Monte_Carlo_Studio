import math
import numpy as np
from rw_mc_studio.random_walk import theoretical_msd,theoretical_mean_radius,simulate_endpoints,summarize_endpoints,step_second_moment,return_probability
from rw_mc_studio.monte_carlo import theoretical_nd_ball_volume,circle_area_mc,nd_ball_volume_mc,grid_sphere_volume,repeated_circle_trials


def test_uniform_second_moment_and_msd():
    assert abs(step_second_moment('uniform',0.5,1.5)-13/12)<1e-12
    assert abs(theoretical_msd(12,'uniform',0.5,1.5)-13)<1e-12

def test_ball_volumes():
    assert abs(theoretical_nd_ball_volume(2)-math.pi)<1e-12
    assert abs(theoretical_nd_ball_volume(3)-4*math.pi/3)<1e-12

def test_fixed_multinomial_endpoint_msd():
    ep,meta=simulate_endpoints(2,500,30000,'fixed',1.0,seed=7,return_metadata=True)
    assert meta['engine']=='multinomial-endpoint'
    msd=summarize_endpoints(ep)['msd']
    assert abs(msd-500)/500<0.03

def test_random_uniform_endpoint_msd():
    ep=simulate_endpoints(2,200,10000,'uniform',0.5,1.5,seed=9,max_block_draws=200000)
    msd=summarize_endpoints(ep)['msd']; theory=theoretical_msd(200,'uniform',0.5,1.5)
    assert abs(msd-theory)/theory<0.05

def test_circle_mc_close():
    assert abs(circle_area_mc(200000,seed=123)['estimate']-math.pi)<0.03

def test_zero_hit_high_d_ci_not_zero_width():
    out=nd_ball_volume_mc(50,1000,seed=1)
    if out['hits_inside']==0:
        assert out['volume_ci_high_exact_binomial']>0

def test_grid_3d_slice_method():
    out=grid_sphere_volume(50)
    assert abs(out['estimate']-4*math.pi/3)<0.05

def test_repeated_mc_semantics():
    out=repeated_circle_trials(1000,30,seed=4)
    assert out['mean_estimate_ci_low']<math.pi<out['mean_estimate_ci_high'] or abs(out['mean_estimate']-math.pi)<0.05

def test_return_probability_interval_bounds():
    out=return_probability(3,max_steps=100,n_trials=500,seed=2)
    assert 0<=out['probability_ci_low']<=out['return_probability_by_horizon']<=out['probability_ci_high']<=1


def test_correct_2d_asymptotic_coefficient():
    c=theoretical_mean_radius(2,10000)/100
    assert abs(c-math.sqrt(math.pi)/2)<1e-12

def test_endpoint_memory_guard():
    import pytest
    with pytest.raises(ValueError):
        simulate_endpoints(200,100,100000,seed=1,max_endpoint_values=1_000_000)


def test_disk_alias_and_semantic_note():
    from rw_mc_studio.monte_carlo import circle_area_mc, disk_area_mc
    a=circle_area_mc(1000,seed=11)
    b=disk_area_mc(1000,seed=11)
    assert a['estimate']==b['estimate']
    assert 'unit disk' in a['semantic_note']

def test_confidence_validation():
    import pytest
    from rw_mc_studio.statistics import clopper_pearson, mean_ci_t
    with pytest.raises(ValueError): clopper_pearson(1,10,1.2)
    with pytest.raises(ValueError): mean_ci_t(0,1,10,0.0)
