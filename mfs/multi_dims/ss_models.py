"""
State-space models used in the experiments.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from mfs.utils import GaussianSumND
from mfs.typings import FloatScalar, JArray, JFloat
from typing import Callable, Union, Tuple, NamedTuple


def satellite_orbital_stability(a=1, b=1, c=1):
    def drift(x: JArray) -> JArray:
        return jnp.array([x[1],
                          -b * x[1] - jnp.sin(x[0]) - c * jnp.sin(2 * x[0])])

    def dispersion(x) -> JArray:
        return jnp.array([[0., 0.],
                          [0., -a * b * x[1] - b * jnp.sin(x[0])]])

    return drift, dispersion


def prey_predator(multi_indices):
    dt = 1e-3
    T = 2000
    ts = jnp.linspace(dt, dt * T, T)

    alp, beta, delta, gamma, sigma = 4., 4., 4., 4., 0.1

    means = jnp.array([[1., 1.],
                       [1., 1.]])
    covs = jnp.array([[[1., 0.],
                       [0., 1.]],
                      [[2., 0.],
                       [0., 2.]]]) * 0.001
    weights = jnp.array([0.5, 0.5])

    gs = GaussianSumND.new(means, covs, weights, multi_indices)

    def drift(x):
        return x * (x[::-1] * jnp.array([-beta, delta]) + jnp.array([alp, -gamma]))

    def dispersion(x):
        return jnp.diag(sigma * x)

    def emission(x):
        return 1 / (1 + jnp.exp(-x ** 3 + 1))

    def measurement_cond_pmf(y, x):
        return jax.scipy.stats.bernoulli.pmf(y, emission(x[0]))

    @jax.jit
    def simulate(key, integration_steps: int = 100):
        key_x0, key_ws, key_ys = jax.random.split(key, 3)

        ddt = dt / integration_steps
        ddws = math.sqrt(ddt) * jax.random.normal(key=key_ws, shape=(T, integration_steps, 2))

        def integration_milstein(carry, elem):
            x = carry
            ddw = elem

            x = x + drift(x) * ddt + sigma * x * ddw + 0.5 * sigma ** 2 * x * (ddw ** 2 - ddt)
            return x, None

        def scan_body(carry, elem):
            x = carry
            dws = elem

            x, _ = jax.lax.scan(integration_milstein, x, dws)
            return x, x

        x0 = gs.sampler(key_x0, 1)[0]
        _, xs = jax.lax.scan(scan_body, x0, ddws)
        ys = jax.random.bernoulli(key_ys, emission(xs[:, 0]), (T,))
        return x0, xs, ys

    return dt, T, ts, gs, drift, dispersion, emission, measurement_cond_pmf, simulate
