"""
Prey-predator experiments.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.multi_dims.ss_models import prey_predator
from mfs.classical_filters_smoothers import bootstrap_filter
from mfs.classical_filters_smoothers.resampling import stratified
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 2D prey-predator particle filtering.')
parser.add_argument('--nparticles', type=int, default=10000, help='Number of particle filter particles.')
parser.add_argument('--trans', type=str, default='euler', help='The method to compute the transition moments. '
                                                               'Options are "euler" and "tme_*".')
parser.add_argument('--st_mc', type=int, default=0, help='The start MC run.')
parser.add_argument('--ed_mc', type=int, help='The end MC run.')
args = parser.parse_args()

d = 2
nparticles, transition = args.nparticles, args.trans

# Multi-indices
multi_indices_dummy = ((0, 0), (0, 1), (1, 0))

# Model
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(multi_indices_dummy)


if 'euler' in transition:
    def state_cond_m_cov(x, _dt):
        return x + drift(x) * _dt, dispersion(x) @ dispersion(x).T * _dt

elif 'tme' in transition:
    def state_cond_m_cov(x, _dt):
        return tme.mean_and_cov(x, _dt, drift, dispersion, order=int(transition.split('_')[-1]))


# The particle filter
def proposal_sampler(x, _key):
    ms, covs = jax.vmap(state_cond_m_cov, in_axes=[0, None])(x, dt)
    return ms + jnp.einsum('...ij,...j->...i', jnp.linalg.cholesky(covs), jax.random.normal(_key, (nparticles, d)))


@jax.jit
def bf(_ys, _key):
    return bootstrap_filter(proposal_sampler, jax.vmap(measurement_cond_pmf, in_axes=[None, 0]),
                            _ys, init_cond.sampler, _key, nparticles, stratified)[0]


# Monte Carlo
for k in range(args.st_mc, args.ed_mc + 1):
    print(f'Particle filter Monte Carlo run {k} / ({args.st_mc} to {args.ed_mc})')

    # Load the random key
    key = jnp.asarray(np.load('rng_keys.npy')[k])

    # Simulate a trajectory and measurements
    _, xs, ys = simulate(key)

    while np.any(np.isnan(xs)):
        key, _ = jax.random.split(key)
        _, xs, ys = simulate(key)

    # Particle filtering
    filename = f'./results/prey_predator_pf/{transition}_mc_{k}.npz'

    key_pf, _ = jax.random.split(key)
    samples = bf(ys, key_pf)

    # Dump errors.
    np.savez_compressed(filename, errs=np.abs(xs - np.mean(samples, axis=1)))
