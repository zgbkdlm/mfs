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
from mfs.classical_filters_smoothers.smc import particle_filter
from mfs.classical_filters_smoothers.resampling import stratified
from jax.config import config

parser = argparse.ArgumentParser(description='Convergence experiment.')
parser.add_argument('--nparticles', type=int, default=100000, help='Number of particle filter particles.')
parser.add_argument('--k', type=int, default=0, help='Which Monte Carlo run.')
parser.add_argument('--maxmc', type=int, default=10000, help='Number of Monte Carlo samples.')
args = parser.parse_args()

nparticles, k, num_mc = args.nparticles, args.k, args.maxmc

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


def proposal_sampler(ancestors, y, _key):
    """This is the variance-optimal proposal.
    """
    K = Sigma / (Sigma + measurement_noise_var)
    m, cov = F * ancestors + K * (y - F * ancestors), Sigma - K * Sigma
    return m + math.sqrt(cov) * jax.random.normal(_key, ancestors.shape)


def proposal_density(x, ancestors, y):
    K = Sigma / (Sigma + measurement_noise_var)
    m, cov = F * ancestors + K * (y - F * ancestors), Sigma - K * Sigma
    return jax.scipy.stats.norm.pdf(x, m, math.sqrt(cov))


def transition_density(x, ancestors):
    return jax.scipy.stats.norm.pdf(x, F * ancestors, math.sqrt(Sigma))


def init_sampler(_key, n: int):
    return mean0 + math.sqrt(var0) * jax.random.normal(_key, (n, 1))


# Particle filter
@jax.jit
def pf(_ys, _key):
    samples = particle_filter(proposal_sampler,
                              proposal_density,
                              transition_density,
                              measurement_cond_pdf,
                              _ys,
                              init_sampler, _key, nparticles, stratified)[:, :, 0]
    return jnp.mean(samples, axis=1), jnp.var(samples, axis=1)


key = jnp.asarray(np.load('rng_keys.npy')[k])
key_xs, key_ys = jax.random.split(key)

print(f'Convergence test with particle filter. MC run {k} / {num_mc - 1}.')

# Simulate a pair of trajectory and measurements
xs = jnp.linalg.cholesky(matern12(ts, ts)) @ jax.random.normal(key_xs, (T,))
ys = xs + math.sqrt(measurement_noise_var) * jax.random.normal(key_ys, (T,))

# True solution from Kalman filter
true_mfs, true_vfs, _ = kf(ys)

# Particle filter
key_pf, _ = jax.random.split(key_ys)
mfs, vfs = pf(ys, key_pf)

# Errors
errs_means = np.abs(mfs - true_mfs)
errs_variances = np.abs(vfs - true_vfs)
# errs_kls = 0.5 * (vfs / true_vfs - 1 + (true_mfs - mfs) ** 2 / true_vfs + np.log(true_vfs / vfs))
errs_kls = 0.5 * (np.exp(np.log(vfs) - np.log(true_vfs)) - 1
                  + (true_mfs - mfs) ** 2 / true_vfs + np.log(true_vfs) - np.log(vfs))

# Dump results
filename = f'./results/convergence/pf_{nparticles}_mc_{k}.npz'
np.savez(filename, errs_means=errs_means, errs_variances=errs_variances, errs_kls=errs_kls)
