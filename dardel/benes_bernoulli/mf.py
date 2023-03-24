"""
Moment filtering of the Benes--Bernoulli model.

Please note that this script is not meant for pedagogy. if you would like to know how the moment filtering works,
please take a look at the examples in the folder `examples`.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from mfs.one_dim.ss_models import benes_bernoulli
from mfs.one_dim.filtering import moment_filter_cms, moment_filter_rms, moment_filter_scms
from mfs.one_dim.moments import sde_cond_moments_tme_normal, sde_cond_moments_tme
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Benes--Bernoulli moment filtering experiment.')
parser.add_argument('--N', type=int, help='Order. 2 N - 1 is the highest moment order.')
parser.add_argument('--tme', type=int, default=3, help='The order of the TME expansion for approximating the '
                                                       'transition moments.')
parser.add_argument('--mode', type=str, default='central', help='Mode of the moment filter. '
                                                                'Options are "raw", "central" (default), and "scaled".')
parser.add_argument('--normal', action='store_true', help='Compute the transition moments by Normal approximation. '
                                                          'This is helpful for stability because the approximated '
                                                          'moments are valid by definition.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

# Simulation setting and the model definition
N, tme_order, mode, maxmc = args.N, args.tme, args.mode, args.maxmc
dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory = benes_bernoulli(N)

# Functions needed for the moment filter
if args.normal:
    sde_cond_rms, sde_cond_cms, sde_cond_scms, state_cond_mean, state_cond_mean_var = sde_cond_moments_tme_normal(
        drift,
        dispersion,
        dt,
        tme_order,
        N)
    _t = '_normal'
else:
    sde_cond_rms, sde_cond_cms, sde_cond_scms, state_cond_mean, state_cond_mean_var = sde_cond_moments_tme(
        drift, dispersion,
        dt, tme_order)
    _t = ''

# Create the moment filter
if mode == 'raw':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_rms(sde_cond_rms, measurement_cond_pmf,
                                 init_cond.rms, _ys)
elif mode == 'central':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_cms(sde_cond_cms, state_cond_mean, measurement_cond_pmf,
                                 init_cond.cms, init_cond.mean, _ys)
elif mode == 'scaled':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_scms(sde_cond_scms, state_cond_mean_var, measurement_cond_pmf,
                                  init_cond.scms, init_cond.mean, jnp.sqrt(init_cond.variance), _ys)
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')

# Monte Carlo
for k in range(maxmc):
    print(f'{N}-order ({mode}) moment filter Monte Carlo run {k} / {maxmc - 1}')

    # Load the random key
    key = jnp.asarray(np.load('rng_keys.npy')[k])
    key_x0, key_xs, key_ys = jax.random.split(key, 3)

    # Simulate a trajectory and measurements
    x0 = init_cond.sampler(key_x0, 1)[0]
    xs = simulate_trajectory(x0, key_xs)
    ys = jax.random.bernoulli(key_ys, logistic(xs), (T,))

    # Moment filtering
    filename = f'./results/benes_bernoulli_mf/{mode}{_t}_N_{N}_mc_{k}.npz'
    if mode == 'raw':
        rmss, nell = moment_filter(ys)
        np.savez_compressed(filename, rmss=rmss, nell=nell)
    elif mode == 'central':
        cmss, means, nell = moment_filter(ys)
        np.savez_compressed(filename, cmss=cmss, means=means, nell=nell)
    else:
        scmss, means, scales, nell = moment_filter(ys)
        np.savez_compressed(filename, scmss=scmss, means=means, scales=scales, nell=nell)
