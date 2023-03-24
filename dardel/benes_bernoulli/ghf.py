"""
Gauss--Hermite filterings of the Benes--Bernoulli model.

Note that the extended Kalman filter will fail for this model by definition.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.one_dim.ss_models import benes_bernoulli
from mfs.classical_filters_smoothers.gfs import sgp_filter
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Benes--Bernoulli Gauss--Hermite filtering.')
parser.add_argument('--tme', type=int, default=3, help='The order of the TME expansion for approximating the '
                                                       'transition moments.')
parser.add_argument('--gh', type=int, default=11, help='The order of the Gauss--Hermite integration.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

# Simulation setting and the model definition
tme_order, gh_order, maxmc = args.tme, args.gh, args.maxmc
dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory = benes_bernoulli(2)


def state_cond_m_cov(x, _dt):
    return tme.mean_and_cov(jnp.atleast_1d(x), _dt, drift, dispersion, order=tme_order)


def measurement_cond_m_cov(x):
    p = logistic(x)
    return jnp.atleast_1d(p), jnp.atleast_2d(p * (1 - p))


# The Gauss--Hermite filter
sgps = SigmaPoints.gauss_hermite(d=1, order=gh_order)


@jax.jit
def ghf(_ys):
    return sgp_filter(state_cond_m_cov, measurement_cond_m_cov, sgps,
                      jnp.atleast_1d(init_cond.mean), jnp.atleast_2d(init_cond.variance), dt,
                      _ys[:, None], const_measurement_cov=False)


# Results containers
ghf_mfs = np.zeros((maxmc, T))
ghf_vfs = np.zeros((maxmc, T))

# Monte Carlo
for k in range(maxmc):
    print(f'Gauss--Hermite Monte Carlo run {k} / {maxmc - 1}')

    # Load the random key
    key = jnp.asarray(np.load('rng_keys.npy')[k])
    key_x0, key_xs, key_ys = jax.random.split(key, 3)

    # Simulate a trajectory and measurements
    x0 = init_cond.sampler(key_x0, 1)[0]
    xs = simulate_trajectory(x0, key_xs)
    ys = jax.random.bernoulli(key_ys, logistic(xs), (T,))

    # Gauss--Hermite filter
    mfs, vfs, _ = ghf(ys)
    ghf_mfs[k], ghf_vfs[k] = mfs[:, 0], vfs[:, 0, 0]

# Dump results
np.savez_compressed(f'./results/benes_bernoulli_ghf/gh_{gh_order}_tme_{tme_order}.npz',
                    ghf_mfs=ghf_mfs, ghf_vfs=ghf_vfs)
