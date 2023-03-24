"""
Compute the filtering solution of the Benes--Bernoulli model by brute force.

To compute the numerical true solution, we need to fix a spatial grid. We select the spatial grid by doing a filtering
and choose the grid bounds by the solution's mean and variance, see the argument `--rule`.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from mfs.one_dim.ss_models import benes_bernoulli
from mfs.one_dim.filtering import moment_filter_cms
from mfs.one_dim.moments import sde_cond_moments_tme_normal
from mfs.classical_filters_smoothers.brute_force import brute_force_filter
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Benes--Bernoulli brute force filtering experiment.')
parser.add_argument('--ngrids', type=int, default=2000, help='Number of spatial grids.')
parser.add_argument('--rule', type=int, default=6, help='mean + rule * scale')
parser.add_argument('--nintegrations', type=int, default=100, help='Number of integration steps between two '
                                                                   'measurement times. Higher the better.')
parser.add_argument('--method', type=str, default='chapman-tme-3', help='The method for making predictions from the '
                                                                        'SDE. The default is "chapman-tme-3".')
parser.add_argument('--k', type=int, default=0, help='Which Monte Carlo run.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

# Simulation setting and the model definition
ngrids, rule, nintegrations, pred_method, k, maxmc = args.ngrids, args.rule, args.nintegrations, args.method, args.k, \
    args.maxmc
dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory = benes_bernoulli(7)

# Functions for computing the transition moments
_, state_cond_central_moments, _, state_cond_mean, _ = sde_cond_moments_tme_normal(drift, dispersion, dt, 3, 7)


# JIT compile functions
@jax.jit
def moment_filter(_ys):
    return moment_filter_cms(state_cond_central_moments, state_cond_mean, measurement_cond_pmf,
                             init_cond.cms, init_cond.mean, _ys)


@jax.jit
def ground_truth_filter(_spatial_grids, _ys):
    return brute_force_filter(drift, dispersion, measurement_cond_pmf,
                              init_cond.pdf(_spatial_grids), _spatial_grids, _ys, dt,
                              integration_steps=nintegrations, pred_method=pred_method)


# Monte Carlo
print(f'Brute force filtering Monte Carlo run {k} / {maxmc - 1}')

# Load the random key
key = jnp.asarray(np.load('rng_keys.npy')[k])
key_x0, key_xs, key_ys = jax.random.split(key, 3)

# Simulate a trajectory and measurements
x0 = init_cond.sampler(key_x0, 1)[0]
xs = simulate_trajectory(x0, key_xs)
ys = jax.random.bernoulli(key_ys, logistic(xs), (T,))

# Moment filtering
cmss, means, _ = moment_filter(ys)

if np.any(np.isnan(cmss)):
    print(f'Moment filtering run {k} diverged.')

# Brute force solution
spatial_lb = jnp.min(means - rule * jnp.sqrt(cmss[:, 2]))
spatial_ub = jnp.max(means + rule * jnp.sqrt(cmss[:, 2]))
spatial_grids = jnp.linspace(spatial_lb, spatial_ub, ngrids)
true_pdfs = ground_truth_filter(spatial_grids, ys)

# Dump results
np.savez_compressed(f'./results/benes_bernoulli_brute_force/mc_{k}.npz',
                    spatial_grids=spatial_grids, true_pdfs=true_pdfs, x0=x0, xs=xs, ys=ys)
