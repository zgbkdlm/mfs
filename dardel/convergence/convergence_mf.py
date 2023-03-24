r"""
Test the convergence on an Ornstein--Uhlenbeck model.

.. math::

    dX(t) = -\frac{1}{\ell} \, X(t) dt + \frac{\sqrt{2} \, \sigma}{\sqrt{\ell}} dW(t),
    X(0) ~ N(m_0, V_0).
"""
import argparse
import math
import jax
import jax.numpy as jnp
import numpy as np
from mfs.one_dim.filtering import moment_filter_cms, moment_filter_rms
from mfs.one_dim.moments import raw_moment_of_normal, central_moment_of_normal
from jax.config import config
from functools import partial

parser = argparse.ArgumentParser(description='Convergence experiment.')
parser.add_argument('--N', type=int, help='Order. 2 N - 1 is the highest moment order.')
parser.add_argument('--mode', type=str, default='central', help='Mode of the filtering routine. '
                                                                'Options are raw and central.')
parser.add_argument('--maxmc', type=int, default=10000, help='Number of Monte Carlo samples.')
args = parser.parse_args()

N, mode, num_mc = args.N, args.mode, args.maxmc

config.update("jax_enable_x64", True)
main_key = jax.random.PRNGKey(666)

# Times
dt = 1e-1
T = 100
ts = jnp.linspace(dt, dt * T, T)
measurement_noise_var = 1.

# SDE model
ell, sigma = 1., 0.5
mean0, var0 = 0., sigma ** 2


def drift(x):
    return -1 / ell * x


def dispersion(_):
    return math.sqrt(2) * sigma / math.sqrt(ell)


# Explicit discretisation of the SDE
F, Sigma = math.exp(-dt / ell), sigma ** 2 * (1 - math.exp(-2 * dt / ell))


def matern12(t1, t2):
    return jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell) * sigma ** 2


def measurement_cond_pdf(y, x):
    """p(y | x)
    """
    return jax.scipy.stats.norm.pdf(y, x, math.sqrt(measurement_noise_var))


def kf(_ys):
    def scan_body(carry, elem):
        mf, vf, nell = carry
        y = elem

        mp = F * mf
        vp = F * vf * F + Sigma

        s = vp + measurement_noise_var
        k = vp / s
        mf = mp + k * (y - mp)
        vf = vp - vp * k
        nell -= jax.scipy.stats.norm.logpdf(y, mp, jnp.sqrt(s))
        return (mf, vf, nell), (mf, vf)

    (*_, nell_ys), (_mfs, _vfs) = jax.lax.scan(scan_body, (mean0, var0, 0.), _ys)
    return _mfs, _vfs, nell_ys


num_moments = 2 * N


@partial(jax.vmap, in_axes=[0, None])
@partial(jax.vmap, in_axes=[None, 0])
def state_cond_raw_moments(x, n):
    """E[X_{k}^n | X_{k-1}]
    """
    cond_mean, cond_var = F * x, Sigma
    list_of_rms = jnp.array([raw_moment_of_normal(cond_mean, cond_var, p) for p in range(num_moments)])
    return list_of_rms[n]


@partial(jax.vmap, in_axes=[0, None, None])
@partial(jax.vmap, in_axes=[None, 0, None])
def state_cond_central_moments(x, n, mean):
    """E[(X_{k} - m_k)^n | X_{k-1}]
    """
    cond_mean, cond_var = F * x, Sigma
    list_of_rms = jnp.array([raw_moment_of_normal(cond_mean - mean, cond_var, p) for p in range(num_moments)])
    return list_of_rms[n]


def state_cond_mean(x):
    return F * x


# Initial condition
rms0 = jnp.array([raw_moment_of_normal(mean0, var0, p) for p in range(num_moments)])
cms0 = jnp.array([central_moment_of_normal(var0, p) for p in range(num_moments)])


# JIT filter routines
@jax.jit
def moment_filter_rms(_ys):
    rmss, _ = moment_filter_rms(state_cond_raw_moments,
                                measurement_cond_pdf,
                                rms0,
                                _ys)
    return rmss[:, 1], rmss[:, 2] - rmss[:, 1] ** 2


@jax.jit
def moment_filter_cms(_ys):
    cmss, means, _ = moment_filter_cms(state_cond_central_moments,
                                       state_cond_mean,
                                       measurement_cond_pdf,
                                       cms0, mean0,
                                       _ys)
    return means, cmss[:, 2]


# Run
errs_means = np.zeros((num_mc, T))
errs_variances = np.zeros((num_mc, T))
errs_kls = np.zeros((num_mc, T))

for k in range(num_mc):
    print(f'Convergence test with ({num_moments}) moments. MC run {k} / {num_mc - 1}.')

    key = jnp.asarray(np.load('rng_keys.npy')[k])
    key_xs, key_ys = jax.random.split(key)

    # Simulate a pair of trajectory and measurements
    xs = jnp.linalg.cholesky(matern12(ts, ts)) @ jax.random.normal(key_xs, (T,))
    ys = xs + math.sqrt(measurement_noise_var) * jax.random.normal(key_ys, (T,))

    # True solution from Kalman filter
    true_mfs, true_vfs, _ = kf(ys)

    # Results from moment filter
    if mode == 'raw':
        mfs, vfs = moment_filter_rms(ys)
    elif mode == 'central':
        mfs, vfs = moment_filter_cms(ys)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented.')

    # Errors
    errs_means[k] = np.abs(mfs - true_mfs)
    errs_variances[k] = np.abs(vfs - true_vfs)
    errs_kls[k] = 0.5 * (np.exp(np.log(vfs) - np.log(true_vfs)) - 1
                         + (true_mfs - mfs) ** 2 / true_vfs + np.log(true_vfs) - np.log(vfs))

# Dump results
filename = f'./results/convergence/mf_{mode}_N_{N}.npz'
np.savez(filename, errs_means=errs_means, errs_variances=errs_variances, errs_kls=errs_kls)
