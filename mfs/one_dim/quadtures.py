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
Moment quadrature.
"""
import math
import jax.numpy as jnp
import jax.scipy.linalg
from mfs.typings import JArray, FloatScalar, JFloat
from mfs.utils import ldl_chol
from typing import Tuple, Callable, Any

__all__ = ['moment_quadrature']


def hankel_indices(n: int) -> Tuple[JArray, JArray]:
    """Make matrix indices for generating a pair of the moment Hankel matrices.

    The generated Hankel matrices should look like these:

    [m_0 m_1 m_2 ... m_{n-1}]
    [m_1 m_2 m_3 ... m_{n+0}]
    [m_2 m_3 m_4 ... m_{n+1}]
    [... ... ... ... ...]
    [m_{n-1} m_{n+0} m_{n+1} ... m_{2(n-1)}]

    [m_1 m_2 m_3 ... m_n]
    [m_2 m_3 m_4 ... m_{n+1}]
    [m_3 m_4 m_5 ... m_{n+2}]
    [... ... ... ... ...]
    [m_n m_{n+1} m_{n+2} ... m_{2n-1}]

    Parameters
    ----------
    n : int
        2 * n - 1 = the order of moments (including the zero-order moment).

    Returns
    -------
    JArray, JArray

    Notes
    -----
    https://github.com/google/jax/issues/3171#issuecomment-631826501.
    """
    inds = jnp.arange(n)[:, None] + jnp.arange(n)[None, :]
    return inds, inds + 1


def _gauss_quadrature_golub_welsch(sms: JArray, mean: FloatScalar = 0., scale: FloatScalar = 1.,
                                   sort_nodes: bool = False) -> Tuple[JArray, JArray]:
    """Implementation of Gauss quadrature of Gene H. Golub and John H. Welsch (1969).
    """
    n = math.floor(sms.shape[0] / 2)

    G = sms[hankel_indices(n)[0]]
    R = jax.lax.linalg.cholesky(G).T

    _r = jnp.diagonal(R)
    betas = _r[1:-1] / _r[:-2]
    alphas = jnp.hstack([R[0, 1] / R[0, 0],
                         jnp.diag(R, k=1)[1:] / _r[1:-1] - jnp.diag(R, k=1)[:-1] / _r[:-2]])
    K = jnp.diag(alphas) + jnp.diag(betas, k=-1) + jnp.diag(betas, k=1)

    eigen_vectors, eigen_vals = jax.lax.linalg.eigh(K, sort_eigenvalues=sort_nodes)

    return eigen_vectors[0, :] ** 2, scale * eigen_vals + mean


def moment_quadrature(ms: JArray, mean: FloatScalar = 0., scale: FloatScalar = 1.,
                      sort_nodes: bool = False,
                      ldl: bool = False) -> Tuple[JArray, JArray]:
    """The moment quadrature by orthonormal polynomials.

    Parameters
    ----------
    ms : JArray (2 n, )
        Moments. The array should look like :code:`[1., m_1, m_2, ... m_{2 n - 1}]` when the moments are raw.
        When the mode is central, the array should look like
        :code:`[1., 0., E[(X - mean)^2], ... E[(X - mean)^{2 n - 1}]]`.
    mean : FloatScalar, default=0
        If this :code:`mean` is not zero, then the quadrature will take :code:`ms` as central moments.
    scale : FloatScalar, default=1.
        Scale the central moments. The default value 1 disables this. To use this, set it be the square root of the
        variance, and here ms[2] = 1.
    sort_nodes : bool, default=False
        Whether sort the quadrature nodes in the ascending order.
    ldl : bool, default=False
        Whether use LDL to replace the Cholesky decomposition for better stability. Please note that this is an
        experimental option, since the LDL decomposition implemented here is not performant.

    Returns
    -------
    JArray, JArray
        Quadrature weights and nodes.

    References
    ----------
    Gene H. Golub and John H. Welsch. Calculation of Gauss quadrature rules, Mathematics of Computation, 1969.

    Juha Sarmavuori and Simo Särkkä. Numerical integration as a finite matrix approximation to multiplication operator,
    Journal of Computational and Applied Mathematics, 2019.

    Notes
    -----
    The eigenvalues and eigenvectors of the matrix K can be computed efficiently, as K is tridiagonal. However, at the
    time of this implementation, JaX does not support this feature yet.
    """
    n = math.floor(ms.shape[0] / 2)

    G_inds, H_inds = hankel_indices(n)
    G, H = ms[G_inds], ms[H_inds]

    R = ldl_chol(G) if ldl else jax.lax.linalg.cholesky(G)
    K = jax.lax.linalg.triangular_solve(R, jax.lax.linalg.triangular_solve(R, H, left_side=True, lower=True),
                                        left_side=False, lower=True, transpose_a=True)

    eigen_vectors, eigen_vals = jax.lax.linalg.eigh(K, sort_eigenvalues=sort_nodes)

    return eigen_vectors[0, :] ** 2, scale * eigen_vals + mean


def make_derivatives(f: Callable, order: int, argnum: int = 0):
    """Return a list of derivatives of f with respect to an argument.

    [f, f', f'', ..., f^{order}]
    """
    derivative = f
    list_of_derivatives = [derivative]
    for _ in range(order):
        def derivative(x, *args, func=derivative):
            return jax.grad(func, argnums=argnum)(x, *args)

        list_of_derivatives.append(derivative)
    return list_of_derivatives


def taylor_quadrature(f: Callable[[FloatScalar, Any], FloatScalar],
                      cms: JArray,
                      mean: FloatScalar,
                      order: int,
                      *operands) -> JFloat:
    r"""Quadrature by Taylor expansion.

    f(x) \approx f(mean) + f'(mean) (x - mean) + 0.5 f''(mean) (x - mean) ** 2 + ...

    Parameters
    ----------
    f : (), ... -> ()
        Scalar to scalar.
    cms : JArray
        Central moments.
    mean : FloatScalar
        Mean.
    order : int
        Order.
    *operands :
        Arguments passed to :code:`f`.

    Returns
    -------
    JArray ()
        E[f(X)].
    """
    derivative_funcs = make_derivatives(f, order)

    result = derivative_funcs[0](mean, *operands)
    for r in range(1, order + 1):
        result += derivative_funcs[r](mean, *operands) * cms[r] / math.factorial(r)
    return result
