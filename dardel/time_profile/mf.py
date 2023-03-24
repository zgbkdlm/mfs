"""
Compute the running time.
"""
import time
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
                                 init_cond.rms, _ys)[0]
elif mode == 'central':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_cms(sde_cond_cms, state_cond_mean, measurement_cond_pmf,
                                 init_cond.cms, init_cond.mean, _ys)[0]
elif mode == 'scaled':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_scms(sde_cond_scms, state_cond_mean_var, measurement_cond_pmf,
                                  init_cond.scms, init_cond.mean, jnp.sqrt(init_cond.variance), _ys)[0]
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')


def simulator(_key):
    key_x0, key_xs, key_ys = jax.random.split(_key, 3)

    # Simulate a trajectory and measurements
    x0 = init_cond.sampler(key_x0, 1)[0]
    xs = simulate_trajectory(x0, key_xs)
    return jax.random.bernoulli(key_ys, logistic(xs), (T,))


# Compute times
elapsed_times = np.zeros((maxmc,))
filename = f'./results/times/mf_{mode}{_t}_N_{N}.npz'

key_nan = jnp.asarray(np.load('rng_keys.npy')[-1])

for k in range(-1, maxmc):
    print(f'{N}-order ({mode}) moment filter Monte Carlo run {k} / {maxmc - 1}')

    key = jnp.asarray(np.load('rng_keys.npy')[k])
    ys = simulator(key)

    # Filtering
    if k == -1:
        print('Compile...')
        result = moment_filter(ys)
    else:
        while True:
            tic = time.time()
            result = moment_filter(ys)
            result.block_until_ready()
            elapse = time.time() - tic

            if ~np.any(np.isnan(result)):
                break
            else:
                key_nan, _ = jax.random.split(key_nan)
                ys = simulator(key_nan)

        elapsed_times[k] = elapse

np.savez_compressed(filename, elapsed_times=elapsed_times)
