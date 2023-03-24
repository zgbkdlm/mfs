"""
Prey-predator experiments.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.multi_dims.ss_models import prey_predator
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from mfs.classical_filters_smoothers.gfs import sgp_filter, ekf
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 2D prey-predator Gauss--Hermite and EKF filtering.')
parser.add_argument('--gh', type=int, default=11, help='The order of the Gauss--Hermite integration.')
parser.add_argument('--trans', type=str, default='euler', help='The method to compute the transition moments. '
                                                               'Options are "euler" and "tme_*".')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

d = 2
gh_order, transition, maxmc = args.gh, args.trans, args.maxmc

# Multi-indices
multi_indices_dummy = ((0, 0), (0, 1), (1, 0))

# Model
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(multi_indices_dummy)
mean0, cov0 = init_cond.mean, init_cond.cov

if 'euler' in transition:
    def state_cond_m_cov(x, _dt):
        return x + drift(x) * _dt, dispersion(x) @ dispersion(x).T * _dt

elif 'tme' in transition:
    def state_cond_m_cov(x, _dt):
        return tme.mean_and_cov(x, _dt, drift, dispersion, order=int(transition.split('_')[-1]))


def measurement_cond_m_cov(x):
    p = emission(x[0])
    return jnp.atleast_1d(p), jnp.atleast_2d(p * (1 - p))


# The Gauss--Hermite filter
sgps = SigmaPoints.gauss_hermite(d=d, order=gh_order)


@jax.jit
def ghf(_ys):
    return sgp_filter(state_cond_m_cov, measurement_cond_m_cov, sgps, mean0, cov0, dt,
                      _ys[:, None], const_measurement_cov=False)


@jax.jit
def extended_kf(_ys):
    return ekf(state_cond_m_cov, measurement_cond_m_cov, mean0, cov0, dt, _ys[:, None])


# Monte Carlo
for k in range(maxmc):
    print(f'Gaussian filter Monte Carlo run {k} / {maxmc - 1}')

    # Load the random key
    key = jnp.asarray(np.load('rng_keys.npy')[k])

    # Simulate a trajectory and measurements
    _, xs, ys = simulate(key)

    while np.any(np.isnan(xs)):
        key, _ = jax.random.split(key)
        _, xs, ys = simulate(key)

    # Gaussian filtering
    filename = f'./results/prey_predator_ghf_ekf/{transition}_mc_{k}.npz'

    mfs, _, _ = ghf(ys)
    errs_ghf = np.abs(xs - mfs)

    mfs, _, _ = extended_kf(ys)
    errs_ekf = np.abs(xs - mfs)

    np.savez_compressed(filename, errs_ghf=errs_ghf, errs_ekf=errs_ekf)
