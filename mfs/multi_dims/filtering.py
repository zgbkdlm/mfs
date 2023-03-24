# Copyright (C) 2022 Zheng Zhao
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Multidimensional moment filters.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from mfs.multi_dims.quadratures import moment_quadrature_nd
from mfs.typings import Array, JArray, JFloat, FloatScalar
from functools import partial
from typing import Callable, Tuple, Union, Any

__all__ = ['moment_filter_nd_rms',
           'moment_filter_nd_cms',
           'moment_filter_nd_scms']


def moment_filter_nd_scms(state_cond_scms: Tuple[Callable[[JArray, JArray, JArray, JArray], JArray], str],
                          state_cond_mean_vars: Callable[[JArray], JArray],
                          measurement_cond_pdf: Callable[[Any, JArray], FloatScalar],
                          ys: JArray,
                          moments_partial_order: Tuple[Array, Array],
                          scms0: JArray,
                          mean0: JArray,
                          scale0: JArray,
                          stable: bool = False) -> Tuple[JArray, JArray, JArray, JFloat]:
    r"""Moment filter for multidimensional states with scaled central moments.

    The filter applies for models

    .. math::

        dX(t) = a(X(t), t) dt + b(X(t), t) dW(t), \\
        Y_k | X(t_k) ~ p(y_k | x_k),

    or with a discrete-time state

    .. math::

        X_k | X_{k-1} ~ p(x_k | x_{k-1}),

    as long as we have access to the conditional moments :math:`E[X_k^n | X_{k-1}]` and the measurement probability
    density function.

    Parameters
    ----------
    state_cond_scms : Tuple[[(..., d), (z, d) or (z, ), (d, ), (d, ) -> (..., z)], str]

        Conditional scaled central moments of the state. That is, given a d-dimensional multi-index
        :math:`(n_1, n_2, ... n_d)\in\mathcal{Z}^d`, and the previous state :math:`X_{k-1}`, the
        conditional scaled central moment :math:`E[X_k^n | X_{k-1}]`is defined by

        .. math::

            E[((X_{k, 1} - m_{k, 1}) / r_1)^{n_1} \, (((X_{k, 2} - m_{k, 2}) / r_2)^{n_2})
            \cdots ((X_{k, d} - m_{k,d}) / r_d)^{n_d} | X_{k-1}],

        where :math:`X_{k, d}` denotes the :math:`d`-th element of the vector :math:`X_k`, and :math:`m_k := E[X_k]` and
        :math:`r_d = \sqrt{E[(X_{k, d} - m_{k, d})^2]}` denote the mean and scale.

        To compute such the moment, we desire a function of signature (d, ), (d, ), (d, ), (d, ) -> float, where the
        first argument is for the previous time state :math:`X_{k-1} \in \mathbb{R}^d`, the second argument is for the
        multi-index :math:`n \in \mathbb{Z}^d`, and the last two arguments are for the mean and scale vectors
        :math:`m_k \mathbb{R}^d`. Now imagine we have :math:`z` many multi-indices, we vectorise the function along the
        second argument to get a signature (d, ), (z, d), (d, ) -> (z, ). Furthermore, suppose that we have ... many
        inputs (e.g., quadrature nodes), then we further vectorise the function signature to
        (..., d), (z, d), (d, ) -> (..., z). This is an example for how to create such a function by using the TME
        method:

        .. code-block::

            @partial(jax.vmap, in_axes=[0, None, None, None])
            @partial(jax.vmap, in_axes=[None, 0, None, None])
            def state_cond_central_moments(x: JArray, multi_index: JArray, mean: JArray, scale: JArray) -> JArray:
                def phi(_x):
                    return jnp.prod(((_x - mean) / scale) ** multi_index)
                return tme.expectation(phi, x, dt, drift, dispersion, order=tme_order)

        However, due to some implementation problems in JAX, it might be hard to vectorise over multi-indices for some
        other methods. For instance, when computing the conditional moments of a Normal distribution, it is hard to
        make multi-index a JAX type (see the docstring of `central_moments_mvn_kan`). A solution is to create a list of
        functions, each of which computes the moment under a specific multi-index. Then combine `jax.lax.switch` and
        the list, we can vmap it. As an example, the following code block shows how to create such a function for
        computing the moments by using the Euler--Maruyama approximation.

        .. code-block::

            multi_indices = ... from the upper context

            @partial(jax.vmap, in_axes=[0, None, None, None])
            @partial(jax.vmap, in_axes=[None, 0, None, None])
            def state_cond_scaled_central_moments(x: JArray, index: JInt, mean: JArray, scale: JArray) -> JArray:
                cond_mean = x + drift(x) * dt
                cond_cov = dispersion(x) @ dispersion(x).T * dt
                cms = jnp.asarray(
                    [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
                s = jnp.prod(scale ** multi_indices_jax[index])
                return cms[index] / s

        where the argument `index` should be `jnp.arange(multi_indices.shape[0])` that maps from 0, 1, 2, ... to the
        multi-indices. But beware that this function is giga-slow to compile.

        You can check `mfs.multi_dims.moments` for more examples how to implement such functions.

        The argument `state_cond_central_moments` is a tuple of a callable function and a string-valued flag. The
        callable function computes the conditional moments, while the flag indicates the signature of the callable
        function. If `flag == "multi-index"`, then the function signature is `(..., d), (z, d), (d, ) -> (..., z)`.
        Otherwise, the signature is (..., d), (z, ), (d, ) -> (..., z).

    state_cond_mean_vars : (..., d) -> (..., d)
        Conditional mean and the diagonal of the covariance.
    measurement_cond_pdf : (...), (d, ) -> float
        The conditional measurement probability density function :math:`p_{Y \mid X}(y \mid x)`. The function's first
        argument is y, the shape of which is determined by the user.
    ys : JArray (T, ...)
        Measurements.
    moments_partial_order : Tuple[Array (z, d), JArray (d + 1, s, s)]
        A tuple of two jax arrays that defines how the moments are ordered. The first array are the moments
        multi-indices of shape (z, d). The second array are integer-indices that used to generate the Gram and Hankel
        matrices. s = comb(N - 1 + d, N - 1).
    scms0 : JArray (z, )
        Initial scaled central moments.
    mean0 : JArray (d, ), default=None
        Initial mean vector.
    scale0 : JArray (d, )
        Initial scale vector.
    stable : bool, default=False
        Set this True will make the moment filter numerically more stable. Currently, this is implemented by an LDL
        moment matrix completion.

    Returns
    -------
    JArray (T, z), JArray (T, d), JArray (T, d), JFloat
        Filtering scaled central moments, means, scales, and the negative log likelihood.

    Notes
    -----
    z is the number of moments which is binom(2 * N - 1 + d, 2 * N - 1) ~ O((2N)^d).

    The API will change in the future, in particular the weird 'index' and 'multi-index' string.
    """
    multi_indices, inds = moments_partial_order

    if multi_indices.shape[0] != scms0.shape[0]:
        raise ValueError(f'The size of multi_indices {multi_indices.shape[0]} '
                         f'must match that of cms0 {scms0.shape[0]}.')

    d = multi_indices.shape[-1]
    _mean_multi_indices = jnp.asarray(np.eye(d, dtype='uint8').tolist())  # This guarantees valid jax integers.

    state_cond_scms_fn, signature = state_cond_scms
    if signature == 'multi-index':
        state_moments_indices = multi_indices
    else:
        state_moments_indices = jnp.arange(multi_indices.shape[0])

    @partial(jax.vmap, in_axes=[None, 0, None, None, None])
    @partial(jax.vmap, in_axes=[None, None, 0, None, None])
    def unnormalised_posterior_moment_integrand(y: FloatScalar, x: JArray, multi_index: Array,
                                                mean: Union[JArray, FloatScalar], scale: Union[JArray, FloatScalar]):
        """Signature after vmap: (), (..., d), (z, d), (d, ), (d, ) -> (..., z).
        """
        return jnp.prod(((x - mean) / scale) ** multi_index) * measurement_cond_pdf(y, x)

    def scan_body_scms(carry, elem):
        scms, mean, scale, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature_nd(scms, inds, mean, scale, ldl=stable)
        # TODO: cond_scale or cond_cov?
        cond_means, cond_vars = state_cond_mean_vars(nodes)
        mean = jnp.einsum('ij,i->j', cond_means, weights)
        scale = jnp.sqrt(jnp.einsum('ij,i->j', cond_vars, weights))
        scms = jnp.einsum('ij,i->j', state_cond_scms_fn(nodes, state_moments_indices, mean, scale), weights)

        # Update step
        weights, nodes = moment_quadrature_nd(scms, inds, mean, scale, ldl=stable)
        pdf_y = jnp.einsum('i,i', jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        mean = jnp.einsum('ij,i->j', unnormalised_posterior_moment_integrand(y, nodes, _mean_multi_indices, 0., 1.),
                          weights) / pdf_y
        scale = jnp.sqrt(
            jnp.einsum('ij,i->j', unnormalised_posterior_moment_integrand(y, nodes, _mean_multi_indices * 2, mean, 1.),
                       weights) / pdf_y)
        scms = jnp.einsum('ij,i->j',
                          unnormalised_posterior_moment_integrand(y, nodes, multi_indices, mean, scale),
                          weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (scms, mean, scale, nell), (scms, mean, scale)

    (*_, nell_ys), (scmss, means, scales) = jax.lax.scan(scan_body_scms, (scms0, mean0, scale0, 0.), ys)
    return scmss, means, scales, nell_ys


def moment_filter_nd_cms(state_cond_central_moments: Tuple[Callable[[JArray, JArray, JArray], JArray], str],
                         state_cond_mean: Callable[[JArray], JArray],
                         measurement_cond_pdf: Callable[[Any, JArray], FloatScalar],
                         ys: JArray,
                         moments_partial_order: Tuple[Array, Array],
                         cms0: JArray,
                         mean0: JArray,
                         stable: bool = False) -> Tuple[JArray, JArray, JFloat]:
    r"""Filtering with central moments.

    Parameters
    ----------
    state_cond_central_moments : Tuple[[(..., d), (z, d) or (z, ), (d, ) -> (..., z)], str]
        Conditional central moments of the state. Almost the same as with `moment_filter_nd_scms`, except that here has
        no scale.
    state_cond_mean : (..., d) -> (..., d)
        Conditional mean
    cms0 : JArray (z, d)
        Initial central moments.
    measurement_cond_pdf, ys, moments_partial_order, mean0, stable : See the docstring of `moment_filter_nd_scms`.

    Returns
    -------
    JArray (T, z), JArray (T, d), JFloat
        Filtering central moments, means, and the negative log likelihood.
    """
    multi_indices, inds = moments_partial_order

    if multi_indices.shape[0] != cms0.shape[0]:
        raise ValueError(f'The size of multi_indices {multi_indices.shape[0]} must match that of cms0 {cms0.shape[0]}.')

    d = multi_indices.shape[-1]
    _mean_multi_indices = jnp.asarray(np.eye(d, dtype='uint8').tolist())  # This guarantees valid jax integers.

    state_cond_cms_fn, signature = state_cond_central_moments
    if signature == 'multi-index':
        state_moments_indices = multi_indices
    else:
        state_moments_indices = jnp.arange(multi_indices.shape[0])

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    @partial(jax.vmap, in_axes=[None, None, 0, None])
    def unnormalised_posterior_moment_integrand(y: FloatScalar, x: JArray, multi_index: Array,
                                                mean: Union[JArray, FloatScalar]):
        """Signature after vmap: (), (..., d), (z, d), (d, ) -> (..., z).
        """
        return jnp.prod((x - mean) ** multi_index) * measurement_cond_pdf(y, x)

    def scan_body_cms(carry, elem):
        cms, mean, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature_nd(cms, inds, mean, ldl=stable)
        cond_means = state_cond_mean(nodes)
        mean = jnp.einsum('ij,i->j', cond_means, weights)
        cms = jnp.einsum('ij,i->j', state_cond_cms_fn(nodes, state_moments_indices, mean), weights)

        # Update step
        weights, nodes = moment_quadrature_nd(cms, inds, mean, ldl=stable)
        pdf_y = jnp.einsum('i,i', jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        mean = jnp.einsum('ij,i->j', unnormalised_posterior_moment_integrand(y, nodes, _mean_multi_indices, 0.),
                          weights) / pdf_y
        cms = jnp.einsum('ij,i->j',
                         unnormalised_posterior_moment_integrand(y, nodes, multi_indices, mean),
                         weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (cms, mean, nell), (cms, mean)

    (*_, nell_ys), (cmss, means) = jax.lax.scan(scan_body_cms, (cms0, mean0, 0.), ys)
    return cmss, means, nell_ys


def moment_filter_nd_rms(state_cond_raw_moments: Tuple[Callable[[JArray, JArray], JArray], str],
                         measurement_cond_pdf: Callable[[Any, JArray], FloatScalar],
                         ys: JArray,
                         moments_partial_order: Tuple[Array, Array],
                         rms0: JArray,
                         stable: bool = False) -> Tuple[JArray, JFloat]:
    r"""Filtering with raw moments.

    Parameters
    ----------
    state_cond_raw_moments : Tuple[[(..., d), (z, d) or (z, ) -> (..., z)], str]
        Conditional raw moments of the state. Almost the same as with `moment_filter_nd_scms`, except that here has
        no mean and scale.
    rms0 : JArray (z, d)
        Initial raw moments.
    measurement_cond_pdf, ys, moments_partial_order, stable : See the docstring of `moment_filter_nd_scms`.

    Returns
    -------
    JArray (T, z), JFloat
        Filtering raw moments and the negative log likelihood.
    """
    multi_indices, inds = moments_partial_order

    if multi_indices.shape[0] != rms0.shape[0]:
        raise ValueError(f'The size of multi_indices {multi_indices.shape[0]} must match that of cms0 {rms0.shape[0]}.')

    d = multi_indices.shape[-1]
    _mean_multi_indices = jnp.asarray(np.eye(d, dtype='uint8').tolist())  # This guarantees valid jax integers.

    state_cond_rms_fn, signature = state_cond_raw_moments
    if signature == 'multi-index':
        state_moments_indices = multi_indices
    else:
        state_moments_indices = jnp.arange(multi_indices.shape[0])

    @partial(jax.vmap, in_axes=[None, 0, None])
    @partial(jax.vmap, in_axes=[None, None, 0])
    def unnormalised_posterior_moment_integrand(y: FloatScalar, x: JArray, multi_index: Array):
        """Signature after vmap: (), (..., d), (z, d) -> (..., z).
        """
        return jnp.prod(x ** multi_index) * measurement_cond_pdf(y, x)

    def scan_body_rms(carry, elem):
        rms, nell = carry
        y = elem

        # Prediction step
        weights, nodes = moment_quadrature_nd(rms, inds, ldl=stable)
        rms = jnp.einsum('ij,i->j', state_cond_rms_fn(nodes, state_moments_indices), weights)

        # Update step
        weights, nodes = moment_quadrature_nd(rms, inds, ldl=stable)
        pdf_y = jnp.einsum('i,i', jax.vmap(measurement_cond_pdf, in_axes=[None, 0])(y, nodes), weights)
        rms = jnp.einsum('ij,i->j',
                         unnormalised_posterior_moment_integrand(y, nodes, multi_indices),
                         weights) / pdf_y
        nell -= jnp.log(pdf_y)
        return (rms, nell), rms

    (_, nell_ys), rmss = jax.lax.scan(scan_body_rms, (rms0, 0.), ys)
    return rmss, nell_ys
