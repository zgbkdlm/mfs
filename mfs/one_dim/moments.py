"""
Common operations on moments for unidimensional random variables.
"""
import math
import jax
import jax.numpy as jnp
import scipy.linalg
import tme.base_jax as tme
from mfs.one_dim.quadtures import moment_quadrature
from mfs.utils import partial_bell
from mfs.typings import JInt, IntScalar, JFloat, FloatScalar, JArray
from functools import partial
from typing import Callable, Tuple


def central_moment_of_normal(variance: FloatScalar, p: int) -> FloatScalar:
    """p-th central moment of Normal.
    """
    if p % 2 == 0:
        scale = jnp.sqrt(variance)
        return scale ** p * scipy.special.factorial2(p - 1, exact=True)
    else:
        return 0.


def raw_moment_of_standard_normal(p: int) -> float:
    r"""p-th raw moment of standard Normal N(0, 1).

    .. math::

        E[X^p] =
        \begin{cases}
            0 & p is odd,\\
            \frac{p!}{2^{p / 2} \, (p / 2)!} & p is even
        \end{cases}

    Double factorial.

    Parameters
    ----------
    p : int
        Moment order.

    Returns
    -------
    float
        E[X^p].
    """
    if p % 2 == 0:
        return math.factorial(p) / (2 ** (p / 2) * math.factorial(int(p / 2)))
    else:
        return 0.


def raw_moment_of_normal(mean: FloatScalar, variance: FloatScalar, p: int) -> FloatScalar:
    """p-th moment of a Normal random variable given mean and variance.
    """
    return sum([math.comb(p, m) * mean ** m * variance ** ((p - m) / 2) * raw_moment_of_standard_normal(p - m)
                for m in range(p + 1)])


def _make_pascal_triangle(s: int) -> JArray:
    """
    Notes
    -----
    pascal() is not a JAX function, hence, cannot be jitted.
    """
    return jnp.asarray(scipy.linalg.pascal(s, kind='lower', exact=True))


def raw_to_central(rms: JArray) -> JArray:
    """Convert raw moments to central moments.
    """
    s = rms.shape[0]
    inds = jnp.arange(s)
    bn_coeffs = _make_pascal_triangle(s)

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def summand(n: JInt, j: JInt) -> JFloat:
        return jax.lax.cond(n >= j,
                            lambda _: bn_coeffs[n, j] * (-1) ** (n - j) * rms[j] * rms[1] ** (n - j),
                            lambda _: 0.,
                            0.)

    return jnp.sum(summand(inds, inds), axis=1)


def central_to_raw(cms: JArray, mean: float) -> JArray:
    """Convert central moments to raw moments.

    Notes
    -----
    This function is not jittable.
    """
    s = cms.shape[0]
    inds = jnp.arange(s)
    bn_coeffs = _make_pascal_triangle(s)

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def summand(n: JInt, j: JInt) -> float:
        return jax.lax.cond(n >= j,
                            lambda _: bn_coeffs[n, j] * cms[j] * mean ** (n - j),
                            lambda _: 0.,
                            0.)

    return jnp.sum(summand(inds, inds), axis=1)


def raw_to_scaled(rms: JArray, scale: FloatScalar = None) -> JArray:
    """Convert E[X^n] to E[((X - mean) / scale)^n],
    where by default the scale here is defined as the square root of the variance.
    """
    if scale is None:
        scale = jnp.sqrt(rms[2] - rms[1] ** 2)
    return raw_to_central(rms) / jnp.array([scale ** n for n in range(rms.shape[0])])


def scaled_to_central(sms: JArray, scale: FloatScalar) -> JArray:
    """Convert E[((X - mean) / scale)^n] to E[(X - mean)^n].
    """
    return sms * jnp.array([scale ** n for n in range(sms.shape[0])])


def sde_cond_moments_tme(drift: Callable, dispersion: Callable, dt: FloatScalar, tme_order: int):
    """Create conditional moments based on TME expansion.
    """

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JFloat, n: JInt) -> JFloat:
        def phi(u):
            return u ** n

        return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JFloat, n: JInt, mean: FloatScalar) -> JFloat:
        def phi(u):
            return (u - mean) ** n

        return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JFloat, n: JInt, mean: FloatScalar, scale: FloatScalar) -> JFloat:
        def phi(u):
            return ((u - mean) / scale) ** n

        return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JFloat) -> JFloat:
        return jnp.squeeze(tme.expectation(lambda u: u, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JFloat) -> Tuple[JFloat, JFloat]:
        cond_m, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
        return jnp.squeeze(cond_m), jnp.squeeze(cond_var)

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def sde_cond_moments_tme_normal(drift: Callable, dispersion: Callable, dt: FloatScalar, tme_order: int, N: int):
    """Create conditional moments based on TME expansion and Normal approximation.
    """
    num_moments = 2 * N

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JFloat, n: JInt) -> JFloat:
        cond_mean, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
        return jnp.array([raw_moment_of_normal(jnp.squeeze(cond_mean), jnp.squeeze(cond_var), p)
                          for p in range(num_moments)])[n]

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JFloat, n: JInt, mean: FloatScalar) -> JFloat:
        cond_mean, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
        return jnp.array([raw_moment_of_normal(jnp.squeeze(cond_mean) - mean, jnp.squeeze(cond_var), p)
                          for p in range(num_moments)])[n]

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JFloat, n: JInt, mean: FloatScalar, scale: FloatScalar) -> JFloat:
        cond_mean, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
        s = jnp.prod(scale ** jnp.arange(num_moments))
        return jnp.array([raw_moment_of_normal(jnp.squeeze(cond_mean) - mean, jnp.squeeze(cond_var), p)
                          for p in range(num_moments)])[n] / s

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JFloat) -> JFloat:
        return jnp.squeeze(tme.expectation(lambda u: u, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JFloat) -> Tuple[JFloat, JFloat]:
        cond_m, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
        return jnp.squeeze(cond_m), jnp.squeeze(cond_var)

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def sde_cond_moments_euler(drift: Callable, dispersion: Callable, dt: FloatScalar, N: int):
    """Create conditional moments based on Euler--Maruyama and Normal approximation.
    """
    num_moments = 2 * N

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JFloat, n: JInt) -> JFloat:
        cond_mean, cond_var = x + drift(x) * dt, dispersion(x) ** 2 * dt
        return jnp.array([raw_moment_of_normal(cond_mean, cond_var, p) for p in range(num_moments)])[n]

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JFloat, n: JInt, mean: FloatScalar) -> JFloat:
        cond_mean, cond_var = x + drift(x) * dt, dispersion(x) ** 2 * dt
        return jnp.array([raw_moment_of_normal(cond_mean - mean, cond_var, p) for p in range(num_moments)])[n]

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JFloat, n: JInt, mean: FloatScalar, scale: FloatScalar) -> JFloat:
        cond_mean, cond_var = x + drift(x) * dt, dispersion(x) ** 2 * dt
        s = jnp.prod(scale ** jnp.arange(num_moments))
        return jnp.array([raw_moment_of_normal(cond_mean - mean, cond_var, p) for p in range(num_moments)])[n] / s

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JFloat) -> JFloat:
        return x + drift(x) * dt

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JFloat) -> Tuple[JFloat, JFloat]:
        return x + drift(x) * dt, dispersion(x) ** 2 * dt

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def sms_to_cumulants(sms: JArray, mean: FloatScalar, scale: FloatScalar) -> JArray:
    r"""Convert scaled central moments to cumulants.

    Note that

    .. math::

        K(z) := \log M(z) = \frac{\mu}{\sigma} \, z + \log S(\sigma \, z),

    where :math:`S` stands for the scaled central moment-generating function, and :math:`\mu, \sigma` are the mean and
    scale, respectively. So the n-th cumulant :math:`k_n` in terms of the scaled central moments, by Faa di Bruno's
    formula is

    .. math::

        k_n = \sum^n_{k=1} (-1)^{k-1} \, (k-1)! \, B_{n,k}(\sigma \, s_1, \sigma^2 \, s_2, \ldots,
                \sigma^{n-k+1} \, s_{n-k+1}),

    where :math:`s`'s are the scaled central moments.

    Parameters
    ----------
    sms : JArray (2 n, )
        Scaled central moments
    mean : FloatScalar
        Mean.
    scale : FloatScalar
        Scale.

    Returns
    -------
    JArray (2 n - 1, )
        Cumulants k_1, ..., k_{2 n - 1}.

    Notes
    -----
    The zero-th cumulant k_0 is ignored, so the return starts from k_1.
    """

    def nth_cumulant(n: int) -> FloatScalar:
        if n == 1:
            return sms[1] if mean == 0. else mean
        elif n == 0:
            raise ValueError(f'The implementation does not allow n = 0.')
        else:
            return sum([(-1) ** (k - 1) * math.factorial(k - 1) * partial_bell(n, k, scaled_to_central(sms, scale)[1:])
                        for k in range(1, n + 1)])

    return jnp.array([nth_cumulant(n) for n in range(1, sms.shape[0])])


def characteristic_fn(z: FloatScalar, ms: JArray, mean: FloatScalar = 0., scale: FloatScalar = 1.) -> JFloat:
    r"""Characteristic function computed by moments.

    .. math::

        \mathbb{E}[\exp(\mathrm{i} \, z \, X)] \approx \sum_n w_n \exp(\mathrm{i} \, z \, x_n)

    Parameters
    ----------
    z : FloatScalar
        The location where the characteristic function is evaluated.
    ms : JArray (2 n, )
        Moments. Can be raw/central/scaled depending on if the mean or scale are given.
    mean : FloatScalar, default=0.
        Mean.
    scale : FloatScalar, default=1.
        Scale

    Returns
    -------
    JFloat
        The characteristic function evaluated at `z`.

    Notes
    -----
    We can improve the estimation by Pade approximation which matches the tails better.
    """
    weights, nodes = moment_quadrature(ms, mean, scale, sort_nodes=False)
    return jnp.dot(jnp.exp(1.j * z * nodes), weights)


def characteristic_from_pdf(z: FloatScalar, ps: JArray, xs: JArray) -> JFloat:
    """Characteristic function computed by the probability density function.
    """
    return jnp.trapz(jnp.exp(1.j * z * xs) * ps, xs)
