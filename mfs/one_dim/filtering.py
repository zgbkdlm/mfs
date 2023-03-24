"""
The moment filters with different modes of moments.
"""
import jax
import jax.numpy as jnp
import warnings
from mfs.one_dim.quadtures import moment_quadrature, taylor_quadrature
from mfs.typings import JArray, FloatScalar, JInt, JFloat
from functools import partial
from typing import Callable, Tuple, Union, Any

__all__ = ['moment_filter_rms',
           'moment_filter_cms',
           'moment_filter_scms']


def moment_filter_rms(state_cond_raw_moments: Callable[[JArray, JArray], JArray],
                      measurement_cond_pdf: Callable[[Any, FloatScalar], JArray],
                      rms0: JArray,
                      ys: JArray,
                      stable: bool = False) -> Tuple[JArray, JFloat]:
    r"""Moment filter for 1D states with raw moment representation.

    Parameters
    ----------
    state_cond_raw_moments : (..., ), (number of moments, ), -> (..., number of moments)
        Conditional raw moments of the state :math:`\mathbb{E}[X_k^n \mid X_{k-1} = x]`. This function takes two arrays
        of shapes `(..., )` and `(number of moments, )` as inputs for the :math:`x` and the moment order, respectively,
        and outputs a matrix of shape `(..., number of moments)`.
    measurement_cond_pdf : Any, Scalar -> Scalar
        The conditional measurement probability density function :math:`p_{Y \mid X}(y \mid x)`. The first and second
        arguments are for the variables :math:`y` and :math:`x`, respectively.
    rms0 : JArray (2 n, )
        Initial raw moments.
    ys : JArray (T, )
        Measurements.
    stable : bool, default=False
        Set this True will make the moment filter numerically more stable. Currently, this is implemented by an LDL
        moment matrix completion.

    Returns
    -------
    JArray (T, 2 n), JFloat
        Filtering raw moments and the negative log likelihood
        :math:`-\log p(y_{1:T}) = -\sum^T_{k=1} \log p(y_k \mid y_{k-1})`.
    """
    num_moments = rms0.shape[0]
    moment_powers = jnp.arange(num_moments)

    if num_moments % 2 != 0:
        warnings.warn(f'The order of moments {num_moments - 1} is not odd.')

    @partial(jax.vmap, in_axes=[None, 0, None])
    @partial(jax.vmap, in_axes=[None, None, 0])
    def unnormalised_posterior_moment_integrand(y: Union[JArray, JFloat], x: JArray, n: JInt):
        return x ** n * measurement_cond_pdf(y, x)

    def scan_body(carry, elem):
        rms, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature(rms, sort_nodes=False, ldl=stable)
        rms = jnp.einsum('ij,i->j', state_cond_raw_moments(nodes, moment_powers), weights)

        # Update step
        weights, nodes = moment_quadrature(rms, sort_nodes=False, ldl=stable)
        pdf_y = jnp.dot(jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        rms = jnp.einsum('ij,i->j', unnormalised_posterior_moment_integrand(y, nodes, moment_powers), weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (rms, nell), rms

    (_, nell_ys), rmss = jax.lax.scan(scan_body, (rms0, 0.), ys)
    return rmss, nell_ys


def moment_filter_cms(state_cond_central_moments: Callable[[JArray, JArray, FloatScalar], JArray],
                      state_cond_mean: Callable[[JArray], Tuple[JArray, JArray]],
                      measurement_cond_pdf: Callable[[Any, FloatScalar], JArray],
                      cms0: JArray,
                      mean0: FloatScalar,
                      ys: JArray,
                      stable: bool = False) -> Tuple[JArray, JArray, JFloat]:
    r"""Moment filter for 1D states with central moment representation.

    Parameters
    ----------
    state_cond_central_moments : (..., ), (number of moments, ), () -> (..., number of moments)
        Conditional central moments of the state :math:`\mathbb{E}[(X_k - m_k)^n \mid X_{k-1} = x]`, where
        :math:`m_k := \mathbb{E}[X_k]`. This function takes two arrays
        of shapes `(..., )` and `(number of moments, )` and a float as inputs for the :math:`x`, the moment order, and
        the mean respectively, and outputs a matrix of shape `(..., number of moments)`.
    state_cond_mean : (..., ) -> (..., ), (..., )
        Conditional mean of the state.
    measurement_cond_pdf : Any, Scalar -> Scalar
        The conditional measurement probability density function :math:`p_{Y \mid X}(y \mid x)`. The first and second
        arguments are for the variables :math:`y` and :math:`x`, respectively.
    cms0 : JArray (2 n, )
        Initial scaled central moments.
    mean0 : FloatScalar
        Initial mean.
    ys : JArray (T, )
        Measurements.
    stable : bool, default=False
        Set this True will make the moment filter numerically more stable. Currently, this is implemented by an LDL
        moment matrix completion.

    Returns
    -------
    JArray (T, 2 n), JArray (T, ), JFloat
        Filtering central moments, means, and the negative log likelihood
        :math:`-\log p(y_{1:T}) = -\sum^T_{k=1} \log p(y_k \mid y_{k-1})`.
    """
    num_moments = cms0.shape[0]
    orders = jnp.arange(num_moments)

    if num_moments % 2 != 0:
        warnings.warn(f'The order of moments {num_moments - 1} is not odd.')

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    @partial(jax.vmap, in_axes=[None, None, 0, None])
    def unnormalised_posterior_moment_integrand(y: FloatScalar, x: JArray, n: JInt, mean: FloatScalar):
        return (x - mean) ** n * measurement_cond_pdf(y, x)

    def scan_body(carry, elem):
        cms, mean, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature(cms, mean, sort_nodes=False, ldl=stable)
        cond_means = state_cond_mean(nodes)
        mean = jnp.dot(cond_means, weights)  # Is einsum better than dot?
        cms = jnp.einsum('ij,i->j', state_cond_central_moments(nodes, orders, mean), weights)

        # Update step
        weights, nodes = moment_quadrature(cms, mean, sort_nodes=False, ldl=stable)
        pdf_y = jnp.dot(jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        mean = jnp.dot(unnormalised_posterior_moment_integrand(y, nodes, jnp.array([1]), 0.)[:, 0], weights) / pdf_y
        cms = jnp.einsum('ij,i->j',
                         unnormalised_posterior_moment_integrand(y, nodes, orders, mean),
                         weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (cms, mean, nell), (cms, mean)

    (*_, nell_ys), (cmss, means) = jax.lax.scan(scan_body, (cms0, mean0, 0.), ys)
    return cmss, means, nell_ys


def moment_filter_scms(
        state_cond_scaled_central_moments: Callable[[JArray, JArray, FloatScalar, FloatScalar], JArray],
        state_cond_mean_var: Callable[[JArray], Tuple[JArray, JArray]],
        measurement_cond_pdf: Callable[[Any, FloatScalar], JArray],
        scms0: JArray,
        mean0: FloatScalar,
        scale0: FloatScalar,
        ys: JArray,
        stable: bool = False) -> Tuple[JArray, JArray, JArray, JFloat]:
    r"""Moment filter for 1D states with scaled central moment representation.

    Parameters
    ----------
    state_cond_scaled_central_moments : (..., ), (number of moments, ), (), () -> (..., number of moments)
        Conditional scaled central moments of the state :math:`\mathbb{E}[((X_k - m_k) / r_k)^n \mid X_{k-1} = x]`,
        where :math:`m_k := \mathbb{E}[X_k]` and :math:`r_k := \sqrt{\mathbb{E}[(X_k - m_k)^2]}`. This function has
        four arguments, and they are for :math:`x`, moment order, mean, and scale, respectively. The function should
        be able to vectorise over the first two arguments.
    state_cond_mean_var : (..., ) -> (..., ), (..., )
        Conditional mean and variance of the state.
    measurement_cond_pdf : Any, Scalar -> Scalar
        The conditional measurement probability density function :math:`p_{Y \mid X}(y \mid x)`. The first and second
        arguments are for the variables :math:`y` and :math:`x`, respectively.
    scms0 : JArray (2 n, )
        Initial scaled central moments.
    mean0 : FloatScalar
        Initial mean.
    scale0 : FloatScalar
        Initial scale.
    ys : JArray (T, )
        Measurements.
    stable : bool, default=False
        Set this True will make the moment filter numerically more stable. Currently, this is implemented by an LDL
        moment matrix completion.

    Returns
    -------
    JArray (T, 2 n), JArray (T, ), JArray (T, ), JFloat
        Filtering scaled central moments, means, scales, and the negative log likelihood
        :math:`-\log p(y_{1:T}) = -\sum^T_{k=1} \log p(y_k \mid y_{k-1})`.
    """
    num_moments = scms0.shape[0]
    orders = jnp.arange(num_moments)

    if num_moments % 2 != 0:
        warnings.warn(f'The order of moments {num_moments - 1} is not odd.')

    @partial(jax.vmap, in_axes=[None, 0, None, None, None])
    @partial(jax.vmap, in_axes=[None, None, 0, None, None])
    def unnormalised_posterior_moment_integrand(y: FloatScalar, x: JArray, n: JInt,
                                                mean: FloatScalar, scale: FloatScalar):
        return ((x - mean) / scale) ** n * measurement_cond_pdf(y, x)

    def scan_body(carry, elem):
        scms, mean, scale, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature(scms, mean, scale, sort_nodes=False, ldl=stable)
        cond_means, cond_vars = state_cond_mean_var(nodes)
        mean, scale = jnp.dot(cond_means, weights), jnp.sqrt(jnp.dot(cond_vars, weights))
        scms = jnp.einsum('ij,i->j', state_cond_scaled_central_moments(nodes, orders, mean, scale), weights)

        # Update step
        weights, nodes = moment_quadrature(scms, mean, scale, sort_nodes=False, ldl=stable)
        pdf_y = jnp.dot(jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        mean = jnp.dot(unnormalised_posterior_moment_integrand(y, nodes, jnp.array([1]), 0., 1.)[:, 0], weights) / pdf_y
        scale = jnp.sqrt(
            jnp.dot(unnormalised_posterior_moment_integrand(y, nodes, jnp.array([2]), mean, 1.)[:, 0], weights) / pdf_y)
        scms = jnp.einsum('ij,i->j',
                          unnormalised_posterior_moment_integrand(y, nodes, orders, mean, scale),
                          weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (scms, mean, scale, nell), (scms, mean, scale)

    (*_, nell_ys), (scmss, means, scales) = jax.lax.scan(scan_body, (scms0, mean0, scale0, 0.), ys)
    return scmss, means, scales, nell_ys

# def moment_filter_taylor(sde_cond_central_moments: Sequence[Callable[[FloatScalar, FloatScalar, FloatScalar],
# FloatScalar]],
#                          cond_pdf_measurement: Callable[[FloatScalar, FloatScalar], FloatScalar],
#                          cms0: JArray,
#                          mean0: FloatScalar,
#                          ys: JArray) -> Tuple[JArray, JArray]:
#     r"""Moment filter for 1D states with Taylor expansion as the moment quadrature.
#
#     Parameters
#     ----------
#     sde_cond_central_moments : Sequence[(), (), () -> ()]
#         A sequence of functions, where the nth function takes x, mean, and scale as inputs, then outputs
#         E[((X_k - mean) / scale)^n | X_{k-1} = x]. In this Taylor quadrature case, we always use scale = 1.
#     cond_pdf_measurement : FloatScalar, FloatScalar -> FloatScalar
#         The conditional measurement probability density function :math:`p_{Y \mid X}(y \mid x)`. The function's first
#         argument is x.
#     cms0 : JArray (2 n, )
#         Initial scaled central moments.
#     mean0 : FloatScalar
#         Initial mean.
#     ys : JArray (T, )
#         Measurements.
#
#     Returns
#     -------
#     JArray (T, 2 n), JArray (T, )
#         Filtering central moments, and means.
#     """
#     num_moments = cms0.shape[0]
#     max_order = num_moments - 1
#     orders = jnp.arange(num_moments)
#
#     if num_moments % 2 != 0:
#         warnings.warn(f'The order of moments {num_moments - 1} is not odd.')
#
#     sde_cms = [lambda cms, mean, fn=sde_cond_central_moments[n]: taylor_quadrature(fn, cms, mean, max_order, mean, 1.)
#                for n in range(num_moments)]
#
#     def unnormalised_update_integrand(x, y, mean, n):
#         return (x - mean) ** n * cond_pdf_measurement(x, y)
#
#     unnormalised_update_integrands = [lambda x, y, mean, n=n: unnormalised_update_integrand(x, y, mean, n)
#                                       for n in range(num_moments)]
#
#     unnormalised_update_cms = [lambda cms, mean, y,
#                                       fn=unnormalised_update_integrands[n]: taylor_quadrature(fn, cms, mean, max_order,
#                                                                                               y, mean)
#                                for n in range(num_moments)]
#
#     @partial(jax.vmap, in_axes=[0, None, None])
#     def prediction_cms(n: JInt, cms: JArray, mean: FloatScalar):
#         return jax.lax.switch(n, sde_cms, cms, mean)
#
#     @partial(jax.vmap, in_axes=[0, None, None, None])
#     def update_unnormalised_cms(n, cms, mean, y):
#         return jax.lax.switch(n, unnormalised_update_cms, cms, mean, y)
#
#     def scan_body(carry, elem):
#         cms, mean = carry
#         y = elem
#
#         # Prediction step
#         mean = taylor_quadrature(sde_cond_central_moments[1], cms, mean, max_order, 0., 1.)
#         cms = prediction_cms(orders, cms, mean)
#
#         # Update step
#         pdf_y = taylor_quadrature(cond_pdf_measurement, cms, mean, max_order, y)
#         mean = taylor_quadrature(unnormalised_update_integrand, cms, mean, max_order, y, 0., 1) / pdf_y
#         cms = update_unnormalised_cms(orders, cms, mean, y) / pdf_y
#
#         return (cms, mean), (cms, mean)
#
#     _, (cms, means) = jax.lax.scan(scan_body, (cms0, mean0), ys)
#     return cms, means
