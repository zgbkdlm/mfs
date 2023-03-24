"""
Compute the running time.
"""
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.one_dim.ss_models import benes_bernoulli
from mfs.classical_filters_smoothers.smc import bootstrap_filter
from mfs.classical_filters_smoothers.resampling import stratified
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Benes--Bernoulli particle filtering experiment.')
parser.add_argument('--tme', type=int, default=3, help='The order of the TME expansion for approximating the '
                                                       'transition moments.')
parser.add_argument('--nparticles', type=int, default=10000, help='Number of particle filter particles.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

# Simulation setting and the model definition
tme_order, nparticles, maxmc = args.tme, args.nparticles, args.maxmc
dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory = benes_bernoulli(2)


def state_cond_m_cov(x, _dt):
    return tme.mean_and_cov(jnp.atleast_1d(x), _dt, drift, dispersion, order=tme_order)


# The particle filter
def proposal_sampler(x, _key):
    ms, covs = jax.vmap(state_cond_m_cov, in_axes=[0, None])(x, dt)
    return jnp.squeeze(ms) + jnp.squeeze(jnp.sqrt(covs)) * jax.random.normal(_key, (nparticles,))


@jax.jit
def bf(_ys, _key):
    return bootstrap_filter(proposal_sampler, measurement_cond_pmf, _ys, init_cond.sampler, _key,
                            nparticles, stratified)[0]


# Compute times
elapsed_times = np.zeros((maxmc, ))
filename = f'./results/times/pf_{nparticles}.npz'

for k in range(-1, maxmc):
    print(f'Particle filter Monte Carlo run {k} / {maxmc - 1}')

    key = jnp.asarray(np.load('rng_keys.npy')[k])
    key_x0, key_xs, key_ys = jax.random.split(key, 3)

    # Simulate a trajectory and measurements
    x0 = init_cond.sampler(key_x0, 1)[0]
    xs = simulate_trajectory(x0, key_xs)
    ys = jax.random.bernoulli(key_ys, logistic(xs), (T,))

    # Particle filter
    key_pf, _ = jax.random.split(key_ys)
    if k == -1:
        print('Compile...')
        samples = bf(ys, key_pf)
    else:
        tic = time.time()
        samples = bf(ys, key_pf)
        samples.block_until_ready()
        elapse = time.time() - tic
        elapsed_times[k] = elapse

np.savez_compressed(filename, elapsed_times=elapsed_times)
