"""
Multidimensional moment quadrature.
"""
import jax
import jax.numpy as jnp
import numpy as np
import itertools
from mfs.utils import lanczos_ritz, ldl_chol
from mfs.typings import JArray
from typing import Tuple, Union
from functools import partial


def nd_cartesian_prod_indices(d: int, n: int) -> np.ndarray:
    """Suppose that we have `d` `n`-dimensional vectors. Now for each `d` we select one element from its `n`-dimensional
    vector, then we get a `d`-dimensional vector. We are interested in computing all such combinations, that
    is, the `d`-Cartesian products of all the `n`-dimensional vectors. The number of combinations is n ** d.

    This function returns a matrix of indices so that one can generate all such combinations by using the indices.

    Parameters
    ----------
    d : int
        The number of Cartesian products.
    n : int
        Dimension.

    Returns
    -------
    np.ndarray (n ** d, d)
    """
    inds = d * [[i for i in range(n)]]
    return np.asarray(tuple(itertools.product(*inds)), dtype='int64')


def nd_cartesian_prod(x: JArray, inds: Union[JArray, np.ndarray] = None) -> JArray:
    """Suppose that we have `d` `n`-dimensional vectors. Now for each `d` we select one element from its `n`-dimensional
    vector, then we eventually get a `d`-dimensional vector. We are interested in computing all such combinations, that
    is, the `d`-Cartesian products of all the `n`-dimensional vectors. The number of combinations is n ** d.

    Parameters
    ----------
    x : JArray (d, n, ...)
        Input array. The ellipsis means whatever it is.
    inds : Array-like (n ** d, ..., d), default=None
        Static indices. If None, this functon computes the indices.

    Returns
    -------
    JArray (n ** d, ..., d)
        All the combinations.
    """
    if inds is None:
        d, n = x.shape[:2]
        inds = nd_cartesian_prod_indices(d, n)

    if x.ndim == 2:
        return jnp.diagonal(x.T[inds], axis1=1, axis2=2)
        # return jnp.take_along_axis(x.T, inds, axis=0)
        # d, n = x.shape[:2]
        # m = inds.shape[0]
        # _d = np.tile(np.arange(d), m)
        # return x[_d, inds.ravel()].reshape(m, d)
    elif x.ndim == 3:
        return jnp.diagonal(jnp.transpose(x, [2, 0, 1])[inds], axis1=1, axis2=2)
        # d, n = x.shape[:2]
        # m = inds.shape[0]
        # _d = np.tile(np.arange(d), m)
        # return x[_d, inds.ravel()].reshape(m, d, n)
    else:
        raise NotImplementedError('For simplicity we only implemented for ndim <= 3. If you wish for higher dimension, '
                                  'please change the transpose tuple accordingly in the code.')


def nd_cartesian_prod_2(x: JArray) -> JArray:
    r"""Let `x` be an array of shape `(d, n)`. This function returns all the combinations of all the `d`-dimensional
    vectors in the Cartesian

    .. math::

        N \times \cdots^d \times N.

    Parameters
    ----------
    x : JArray (d, n)
        Input array.

    Returns
    -------
    JArray (n ** d, d)
        All the combinations of all the `d`-dimensional vectors.

    Notes
    -----
    Not used. Used for comparison with a legacy implementation.
    """
    d = x.shape[0]
    return jnp.array(jnp.meshgrid(*x)).reshape(-1, d)


def _chained_inner_products(x: JArray):
    return jnp.prod(jax.vmap(jnp.dot, in_axes=[1, 1])(x[:, :-1], x[:, 1:]))


def moment_quadrature_nd(ms: JArray, inds: Union[JArray, np.ndarray],
                         mean: JArray = None,
                         scale: JArray = None,
                         sort_nodes: bool = False,
                         ldl: bool = False) -> Tuple[JArray, JArray]:
    """Multidimensional Gauss quadrature.

    Parameters
    ----------
    ms : JArray (z, )
        Moments, graded lexicographical ordered. The mode of the moments depend on whether `mean` and `scale` are given.
    inds : Array-like (d + 1, )
        Matrices of indices that generate the Gram and Hankel matrices from `ms`. See
        `gram_and_hankel_indices_graded_lexico`.
    mean : JArray (d, ), default=None
        The mean vector. If not None, then `ms` are central moments.
    scale : JArray (d, ), default=None
        The scale vector. If not None, then `ms` are scaled central moments.
    sort_nodes : bool, default=False
        Whether sort the nodes in the ascending order.
    ldl : bool, default=False
        Whether use LDL to replace the Cholesky decomposition for stability. Please note that this is an
        experimental option, since the LDL decomposition implemented here is not performant.

    Returns
    -------
    JArray (r, ), JArray (r, d)
        Quadrature weights and nodes.
    """
    d, n = inds.shape[0] - 1, inds.shape[1]

    G = ms[inds[0]]
    Hs = ms[inds[1:]]

    R = ldl_chol(G) if ldl else jax.lax.linalg.cholesky(G)

    @partial(jax.vmap, in_axes=[0])
    def orthonormalisation(H):
        return jax.lax.linalg.triangular_solve(R, jax.lax.linalg.triangular_solve(R, H, left_side=True, lower=True),
                                               left_side=False, lower=True, transpose_a=True)

    Ks = orthonormalisation(Hs)

    eigvectors, eigvals = jax.lax.linalg.eigh(Ks, sort_eigenvalues=sort_nodes)

    combs_inds = nd_cartesian_prod_indices(d, n)
    combs_eigvals = nd_cartesian_prod(eigvals, combs_inds)
    combs_eigvectors = nd_cartesian_prod(eigvectors, combs_inds)

    weights = jnp.prod(jnp.einsum('ijk,ijk->ik', combs_eigvectors[:, :, :-1], combs_eigvectors[:, :, 1:]), axis=1) \
              * combs_eigvectors[:, 0, 0] * combs_eigvectors[:, 0, -1]

    if mean is None:
        return weights, combs_eigvals
    else:
        if scale is None:
            return weights, combs_eigvals + mean
        else:
            return weights, combs_eigvals * scale + mean

# def gauss_quadrature_2d_lanczos(ms: JArray, inds: Union[JArray, np.ndarray], lanczos_iters: int, mean: JArray = None,
#                                 sort_nodes: bool = False) -> Tuple[JArray, JArray]:
#     d, n = inds.shape[0] - 1, inds.shape[1]
#     e1 = [0.] * inds.shape[1]
#     e1[0] = 1.
#     e1 = jnp.array(e1)
#
#     G = ms[inds[0]]
#     Hs = ms[inds[1:]]
#
#     R = jax.lax.linalg.cholesky(G)
#
#     @partial(jax.vmap, in_axes=[0])
#     def orthonormalisation(H):
#         return jax.lax.linalg.triangular_solve(R, jax.lax.linalg.triangular_solve(R, H, left_side=True, lower=True),
#                                                left_side=False, lower=True, transpose_a=True)
#
#     Ks = orthonormalisation(Hs)
#
#     @partial(jax.vmap, in_axes=[0, None])
#     def lanczos(mat, b):
#         return lanczos_ritz(mat, b, lanczos_iters, sort_eigenvalues=sort_nodes)
#
#     ritz_vals, ritz_vecs = lanczos(Ks, e1)
#     return None
