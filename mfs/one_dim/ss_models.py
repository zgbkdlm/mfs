"""
Test models used in the paper.
"""
import jax
import jax.numpy as jnp
import tme.base_jax as tme
from mfs.utils import GaussianSum1D, simulate_sde


def benes_bernoulli(N: int = 2):
    """The Benes--Bernoulli model.
    """
    dt = 1e-2
    T = 100
    ts = jnp.linspace(dt, dt * T, T)

    init_cond = GaussianSum1D.new(means=jnp.array([-0.5, 0.5]),
                                  variances=jnp.array([0.05, 0.05]),
                                  weights=jnp.array([0.5, 0.5]),
                                  N=N)

    def drift(x):
        return jnp.tanh(x)

    def dispersion(_):
        return 1.

    def logistic(x):
        return 1 / (1 + jnp.exp(-x ** 3 / 5))

    def measurement_cond_pmf(y, x):
        return jax.scipy.stats.bernoulli.pmf(y, logistic(x))

    @jax.jit
    def simulate_trajectory(_x0, _key):
        def m_and_cov(x, _dt):
            return tme.mean_and_cov(jnp.atleast_1d(x), _dt, drift, dispersion, order=3)

        return simulate_sde(m_and_cov, _x0, dt, T, _key, diagonal_cov=False, integration_steps=100)[:, 0]

    return dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory


def well_poisson(true_p1, N: int = 2):
    """The Well-Poisson model for parameter estimation.
    """
    dt = 1e-2
    T = 1000
    ts = jnp.linspace(dt, dt * T, T)

    init_cond = GaussianSum1D.new(means=jnp.array([-0.5, 0.5]),
                                  variances=jnp.array([0.05, 0.05]),
                                  weights=jnp.array([0.5, 0.5]),
                                  N=N)

    def drift(x, p):
        return x * (1 - p * x ** 2)

    def _drift(x):
        return drift(x, true_p1)

    def dispersion(_):
        return 1.

    def emission(x, p):
        return jnp.log(1. + jnp.exp(p * x))

    def measurement_cond_pmf(y, x, p):
        return jax.scipy.stats.poisson.pmf(y, emission(x, p))

    @jax.jit
    def simulate_trajectory(_x0, _key):
        def m_and_cov(x, _dt):
            return tme.mean_and_cov(jnp.atleast_1d(x), _dt, _drift, dispersion, order=3)

        return simulate_sde(m_and_cov, _x0, dt, T, _key, diagonal_cov=False, integration_steps=100)[:, 0]

    return dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate_trajectory
