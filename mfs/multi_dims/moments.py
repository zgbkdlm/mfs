"""
Manipulations for multidimensional moments.
"""
import math
import itertools
import numpy as np
import jax
import jax.numpy as jnp
import tme.base_jax as tme
from scipy.special import comb as scipy_comb
from scipy.special import factorial as scipy_factorial
from typing import Sequence, Callable, Tuple, Union
from mfs.multi_dims.multi_indices import find_indices
from functools import partial
from mfs.typings import JArray, JFloat, Array, FloatScalar, JInt


def _gradient_wrt_multi_index(func: Callable, multi_index: Union[Array, Sequence[int]]) -> Callable[[JArray], JFloat]:
    r"""Multivariate derivative with respect to a multi-index.

    .. math::

        \partial^{|n|} f(x) / \partial x_1^{n_1} \partial x_2^{n_2} \cdots \partial x_d^{n_d}

    Notes
    -----
    Giga-slow when the derivatives order are high.
    """
    derivative = func
    for k, diff_order in enumerate(multi_index):
        for _ in range(diff_order):
            def derivative(x, f=derivative, order=k):
                return jax.grad(f)(x)[order]
    return derivative


def raw_moments_mvn_mgf(mean: Array, cov: Array, multi_index: Union[Array, Sequence[int]]) -> JFloat:
    """Compute multi-index moment from moment-generating function.

    E[X^n] = the n multi-index partial derivative of E[e^{z X}] w.r.t. z evaluated at z = 0.

    See the docstring of `raw_moments_mvn_kan`.
    """

    def mgf(z):
        return jnp.exp(jnp.dot(z, mean) + 0.5 * jnp.dot(z, cov @ z))

    return _gradient_wrt_multi_index(mgf, multi_index)(jnp.zeros((cov.shape[0])))


def central_moments_mvn_kan(cov: Array, multi_index: Sequence[int]) -> FloatScalar:
    """Kan--Magnus method for computing any central moments of multivariate Normal random variables.

    Parameters
    ----------
    cov : Array (d, d)
        A `d` by `d` covariance matrix. Can be either np.ndarray or jax.Array.
    multi_index : Sequence (d, )
        A sequence (e.g., tuple or np.ndarray) of `d` integers.

    Returns
    -------
    FloatScalar
        E[X^n], where n is multi-index from the input, and `X ~ N(0, cov)`.

    References
    ----------
    Raymond Kan. From moments of sum to moments of product. Journal of Multivariate Analysis, 2008. Proposition 1.

    Notes
    -----
    The function applies to both JAX and NumPy `cov` inputs.

    I don't know how to accept `multi_index` as a JAX type, since we need arrays, the shapes of which depend on the
    concrete values of `multi_index`.
    """
    s = sum(multi_index)
    ranges = [tuple(range(sn + 1)) for sn in multi_index]

    vs = np.asarray(tuple(itertools.product(*ranges)), dtype='int64')
    hs = np.asarray(multi_index) / 2 - vs

    signs = (-1) ** np.sum(vs, axis=1)
    combs = np.prod(scipy_comb(np.asarray(multi_index), vs), axis=1)
    quad_prods = (hs[:, None, :] @ cov @ hs[:, :, None] / 2).reshape((vs.shape[0],)) ** (s / 2)

    if s % 2 == 0:
        if isinstance(cov, np.ndarray):
            return np.einsum('i,i,i', signs, combs, quad_prods) / math.factorial(int(s / 2))
        else:
            return jnp.einsum('i,i,i', signs, combs, quad_prods) / math.factorial(int(s / 2))
    else:
        return 0.


def raw_moments_mvn_kan(mean: Array, cov: Array, multi_index: Sequence[int]) -> FloatScalar:
    """Kan--Magnus method for computing any moments of multivariate Normal random variables. Compared to the function
    `central_moments_mvn_kan`, this function allows for non-central moments.

    Parameters
    ----------
    mean : Array (d, )
        A mean vector. Can be either np.ndarray or jax.Array.
    cov : Array (d, d)
        A `d` by `d` covariance matrix. Can be either np.ndarray or jax.Array.
    multi_index : Sequence (d, )
        A sequence (e.g., tuple or np.ndarray) of `d` integers. The function is not jittable in this argument.

    Returns
    -------
    FloatScalar
        E[X^n], where n is multi-index from the input, and `X ~ N(mean, cov)`.

    References
    ----------
    Raymond Kan. From moments of sum to moments of product. Journal of Multivariate Analysis, 2008. Proposition 2.

    Notes
    -----
    The function return is trivially multiplied with `1.` for type check, since np.einsum returns array instead of
    float, while in this case it is clear that the return is a float.
    """
    s = sum(multi_index)
    ranges = [tuple(range(sn + 1)) for sn in multi_index] + [tuple(range(int(s / 2) + 1))]

    vs_and_r = np.asarray(tuple(itertools.product(*ranges)), dtype='int64')
    vs = vs_and_r[:, :-1]
    rs = vs_and_r[:, -1]
    hs = np.asarray(multi_index) / 2 - vs

    signs = (-1) ** np.sum(vs, axis=1)
    combs = np.prod(scipy_comb(np.asarray(multi_index), vs), axis=1)
    quad_prods = (hs[:, None, :] @ cov @ hs[:, :, None] / 2).ravel() ** rs \
                 * (hs @ mean) ** (s - 2 * rs) / (scipy_factorial(rs, exact=True) * scipy_factorial(s - 2 * rs,
                                                                                                    exact=True))
    if isinstance(cov, np.ndarray):
        return np.einsum('i,i,i', signs, combs, quad_prods) * 1.
    else:
        return jnp.einsum('i,i,i', signs, combs, quad_prods)


def moments_nd_uniform(bounds: Sequence[Tuple[float, float]],
                       multi_index: Sequence[int],
                       means: Sequence[float] = None) -> float:
    """Raw moments of multivariate (independent) uniform distribution on any hypercube.

    Parameters
    ----------
    bounds : Sequence/Array-like (d, 2)
        A sequence/array of tuples that encodes the lower and upper bounds of the distribution in the dimensions. E.g.,
        [(-1., 2.), (-2., 3.)] gives a 2D uniform distribution, and the first variable has bounds -1. and 2.
    multi_index : Sequence[int]
        A sequence (e.g., tuple or np.ndarray) of `d` integers.
    means : Sequence/Array-like (d, 2), default=None
        The mean vector. If is `None`, then the mean is zeros.

    Returns
    -------
    float
        The raw moment.
    """
    if means is None:
        means = [0.] * len(bounds)
    items = [((bound[1] - mean) ** (power + 1) - (bound[0] - mean) ** (power + 1)) / (power + 1) / (bound[1] - bound[0])
             for (power, bound, mean) in zip(multi_index, bounds, means)]
    return np.prod(np.array(items)) * 1.


def extract_moments(ms: Array, multi_index: Sequence[int]) -> Array:
    """Extract the moment (based on its multi-index) from a collection of moments (graded lexicographical ordered).

    Parameters
    ----------
    ms : (..., m)
        A collection of moments.
    multi_index : Sequence[int]
        Multi-index.

    Returns
    -------
    Array (...)
        The moment specified by the multi-index.
    """
    return ms[..., find_indices(multi_index)]


def extract_mean(rms: Array, d: int) -> np.ndarray:
    """Extract the mean vector from a collection of raw moments (graded lexicographical ordered).

    Parameters
    ----------
    rms : (..., m)
        A collection of raw moments.
    d : int
        The dimension of the variable.

    Returns
    -------
    np.ndarray (..., d)
        The mean vector.
    """
    if rms.ndim > 1:
        mean = np.zeros((*rms.shape[:-1], d))
    else:
        mean = np.zeros((d,))
    for i in range(d):
        multi_index = [0] * d
        multi_index[i] = 1
        mean[..., i] = extract_moments(rms, multi_index)
    return mean


def extract_cov(ms: Array, d: int) -> np.ndarray:
    """Extract the covariance matrix from a collection of central moments (graded lexicographical ordered). If the
    input is raw moments, then this outputs the second moment matrix instead.

    Parameters
    ----------
    ms : (..., m)
        A collection of central moments or raw moments.
    d : int
        The dimension of the variable.

    Returns
    -------
    np.ndarray (..., d, d)
        The covariance matrix (for central moments input) or the second moment matrix (for raw moments input).
    """
    if ms.ndim > 1:
        cov = np.zeros((*ms.shape[:-1], d, d))
    else:
        cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            multi_index = [0] * d
            multi_index[i] = 1
            multi_index[j] += 1
            cov[..., i, j] = extract_moments(ms, multi_index)
    return cov


def sde_cond_moments_euler_maruyama(drift: Callable,
                                    dispersion: Callable,
                                    dt: FloatScalar,
                                    multi_indices: Union[np.ndarray, Sequence[int]]):
    """Generate the required conditional moments functions for the moment filtering algorithm by using Euler--Maruyama
    approximation.

    Parameters
    ----------
    drift : (d, ) -> (d, )
        SDE drift function.
    dispersion : (d, ) -> (d, w)
        SDE dispersion function.
    dt : FloatScalar
        Time interval. Can be `float` or `jax float`.
    multi_indices : Sequence (d, )
        A sequence (e.g., tuple or np.ndarray) of `d` integers. This implementation does not allow it to be
        `jax.ndarray`.

    Returns
    -------
    Callable, Callable, Callable, Callable, Callable
        The functions that compute

            * conditional raw moments,
            * conditional central moments,
            * conditional scaled central moments,
            * conditional mean,
            * conditional mean and covariance diagonal.

    Notes
    -----
    This implementation is too slow, compared to e.g., tme without using Normal approximation. Need to improve! I think
    the reason is that the `vmap` here does not parallelise for some reasons.
    """
    multi_indices_jax = jnp.asarray(multi_indices)

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JArray, index: JInt) -> JArray:
        cond_mean = x + drift(x) * dt
        cond_cov = dispersion(x) @ dispersion(x).T * dt
        rms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean, cond_cov, multi_index) for multi_index in multi_indices])
        return rms[index]

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JArray, index: JInt, mean: JArray) -> JArray:
        r"""
        .. math::

            \mathbb{E}[(X_k - m_k)^{z_n} \mid X_{k-1}]
        """
        cond_mean = x + drift(x) * dt
        cond_cov = dispersion(x) @ dispersion(x).T * dt
        cms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
        return cms[index]

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JArray, index: JInt, mean: JArray, scale: JArray) -> JArray:
        cond_mean = x + drift(x) * dt
        cond_cov = dispersion(x) @ dispersion(x).T * dt
        cms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
        s = jnp.prod(scale ** multi_indices_jax[index])
        return cms[index] / s

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JArray) -> JArray:
        return x + drift(x) * dt

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JArray) -> Tuple[JArray, JArray]:
        b = dispersion(x)
        return x + drift(x) * dt, jnp.diagonal(b @ b.T) * dt

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def sde_cond_moments_tme_normal(drift: Callable,
                                dispersion: Callable,
                                dt: FloatScalar,
                                tme_order: int,
                                multi_indices: Union[np.ndarray, Sequence[int]]):
    """Generate the required conditional moments functions for the moment filtering algorithm by approximating the
    transition distribution as Normal and approximating the mean and covariance by TME.

    Parameters
    ----------
    drift : (d, ) -> (d, )
        SDE drift function.
    dispersion : (d, ) -> (d, w)
        SDE dispersion function.
    dt : FloatScalar
        Time interval. Can be `float` or `jax float`.
    tme_order : int,
        Order of the TME expansion. When `tme_order = 1`, the expansion is equivalent to Euler--Maruyama (in terms of
        mean and covariance).
    multi_indices : Sequence (d, )
        A sequence (e.g., tuple or np.ndarray) of `d` integers. This implementation does not allow it to be
        `jax.ndarray`.

    Returns
    -------
    Callable, Callable, Callable, Callable, Callable
        The functions that compute

            * conditional raw moments,
            * conditional central moments,
            * conditional scaled central moments,
            * conditional mean,
            * conditional mean and covariance diagonal.
    """
    multi_indices_jax = jnp.asarray(multi_indices)

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JArray, index: JInt) -> JArray:
        cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion, order=tme_order)
        rms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean, cond_cov, multi_index) for multi_index in multi_indices])
        return rms[index]

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JArray, index: JInt, mean: JArray) -> JArray:
        cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion, order=tme_order)
        cms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
        return cms[index]

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JArray, index: JInt, mean: JArray, scale: JArray) -> JArray:
        cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion, order=tme_order)
        cms = jnp.asarray(
            [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
        s = jnp.prod(scale ** multi_indices_jax[index])
        return cms[index] / s

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JArray) -> JArray:
        return tme.expectation(lambda u: u, x, dt, drift, dispersion, order=tme_order)

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JArray) -> Tuple[JArray, JArray]:
        cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion, order=tme_order)
        return cond_mean, jnp.diagonal(cond_cov)

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def sde_cond_moments_tme(drift: Callable,
                         dispersion: Callable,
                         dt: FloatScalar,
                         tme_order: int):
    """Generate the required conditional moments functions for the moment filtering algorithm by using the TME
    approximation. This routine is different from `sde_cond_central_moments_tme_normal` that this does not approximate
    the transition distribution by Normal. Please refer to the paper for details.

    Parameters
    ----------
    drift : (d, ) -> (d, )
        SDE drift function.
    dispersion : (d, ) -> (d, w)
        SDE dispersion function.
    dt : FloatScalar
        Time interval. Can be `float` or `jax float`.
    tme_order : int,
        Order of the TME expansion. When `tme_order = 1`, the expansion is equivalent to Euler--Maruyama.

    Returns
    -------
    Callable, Callable, Callable, Callable, Callable
        The functions that compute

            * conditional raw moments,
            * conditional central moments,
            * conditional scaled central moments,
            * conditional mean,
            * conditional mean and covariance diagonal.
    """

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def state_cond_raw_moments(x: JArray, multi_index: JArray) -> JArray:
        def phi(_x):
            return jnp.prod(_x ** multi_index)

        return tme.expectation(phi, x, dt, drift, dispersion, order=tme_order)

    @partial(jax.vmap, in_axes=[0, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None])
    def state_cond_central_moments(x: JArray, multi_index: JArray, mean: JArray) -> JArray:
        def phi(_x):
            return jnp.prod((_x - mean) ** multi_index)

        return tme.expectation(phi, x, dt, drift, dispersion, order=tme_order)

    @partial(jax.vmap, in_axes=[0, None, None, None])
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def state_cond_scaled_central_moments(x: JArray, multi_index: JArray, mean: JArray, scale: JArray) -> JArray:
        def phi(_x):
            return jnp.prod(((_x - mean) / scale) ** multi_index)

        return tme.expectation(phi, x, dt, drift, dispersion, order=tme_order)

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean(x: JArray) -> JArray:
        return tme.expectation(lambda u: u, x, dt, drift, dispersion, order=tme_order)

    @partial(jax.vmap, in_axes=[0])
    def state_cond_mean_var(x: JArray) -> Tuple[JArray, JArray]:
        cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion, order=tme_order)
        return cond_mean, jnp.diagonal(cond_cov)

    return state_cond_raw_moments, state_cond_central_moments, state_cond_scaled_central_moments, state_cond_mean, \
        state_cond_mean_var


def marginalise_moments(ms: Array, d: int, N: int, var_axis: int) -> Array:
    """Let X1, X2, ... Xd be joint random variables with joint moments `ms`, and suppose that these moments are graded
    lexicographical ordered. This function finds the marginal moments for any X in X1, X2, ..., Xd.

    Parameters
    ----------
    ms : Array (..., z)
        The raw/central/scaled moments. Can be JAX array or np.ndarray.
    d : int
        Dimension.
    N : int
        2 N - 1 is the highest degree of the moments.
    var_axis : int
        The axis=0,1,2,... of the variable that you want to marginalise.

    Returns
    -------
    Array (..., 2 N)
        The marginal moments of the selected random variable.
    """
    multi_indices = np.zeros((2 * N, d), dtype='int64')
    multi_indices[:, var_axis] = np.arange(2 * N)
    return ms[..., find_indices(multi_indices)]
