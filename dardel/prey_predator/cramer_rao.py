"""
Not used.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.multi_dims.ss_models import prey_predator
from mfs.utils import posterior_cramer_rao
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 2D prey-predator Cramer--Rao lower bound.')
parser.add_argument('--maxmc', type=int, default=100000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

multi_indices_dummy = ((0, 0), (0, 1), (1, 0))
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(multi_indices_dummy)


def m_and_cov(x, _dt):
    return tme.mean_and_cov(x, dt, drift, dispersion, order=2)


def logpdf_transition(xt, xs):
    m, cov = m_and_cov(xs, dt)
    return jax.scipy.stats.multivariate_normal.logpdf(xt, m, cov)


def logpdf_likelihood(yt, xt):
    return jax.scipy.stats.bernoulli.logpmf(yt, emission(xt[0]))


keys = jnp.asarray(np.load('rng_keys.npy'))[:args.maxmc]
x0s, xss, yss = jax.vmap(simulate)(keys)
xss = jnp.concatenate([x0s[:, None, :], xss], axis=1)
xss = jnp.transpose(xss, [1, 0, 2])
yss = jnp.transpose(yss, [1, 0])


# Initial Fisher information j0
@jax.vmap
def init_fisher(x):
    return jax.hessian(init_cond.logpdf)(x)


j0 = -jnp.mean(init_fisher(x0s), axis=0)
js = posterior_cramer_rao(xss, yss, j0, logpdf_transition, logpdf_likelihood)
crlbs = jnp.linalg.inv(js)

np.savez_compressed('./results/prey_predator_cramer_rao/crlbs.npz', crlbs=crlbs)
