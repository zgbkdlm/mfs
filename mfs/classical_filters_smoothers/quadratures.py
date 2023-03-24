# Copyright (C) 2021 Zheng Zhao. This module is taken and modified from https://github.com/spdes/chirpgp.
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
Some integration methods, for example, Runge--Kutta and sigma-points.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from mfs.typings import JArray
from typing import Callable, NamedTuple, Union, List, Tuple
from functools import partial

__all__ = ['rk4_m_cov',
           'rk4_m_cov_backward',
           'SigmaPoints',
           'gaussian_expectation']


def rk4_m_cov(m_cov_ode: Callable[[JArray, JArray], Tuple[JArray, JArray]],
              m: JArray, v: JArray, dt: float) -> Tuple[JArray, JArray]:
    """Ad-hoc Runge--Kutta 4 for solving the mean and cov filtering ODE system.

    Parameters
    ----------
    m_cov_ode
    m
    v
    dt

    Returns
    -------

    """
    k1_m, k1_v = m_cov_ode(m, v)
    k2_m, k2_v = m_cov_ode(m + dt * k1_m / 2, v + dt * k1_v / 2)
    k3_m, k3_v = m_cov_ode(m + dt * k2_m / 2, v + dt * k2_v / 2)
    k4_m, k4_v = m_cov_ode(m + dt * k3_m, v + dt * k3_v)
    return m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6, \
           v + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6


def rk4_m_cov_backward(m_cov_ode: Callable[[JArray, JArray, JArray, JArray], Tuple[JArray, JArray]],
                       m: JArray, v: JArray,
                       mf: JArray, vf: JArray, dt: float) -> Tuple[JArray, JArray]:
    """Ad-hoc Runge--Kutta 4 for solving the mean and cov smoothing ODE system.

    Parameters
    ----------
    m_cov_ode
    m
    v
    mf
    vf
    dt

    Returns
    -------

    """
    k1_m, k1_P = m_cov_ode(m, v, mf, vf)
    k2_m, k2_P = m_cov_ode(m + dt * k1_m / 2, v + dt * k1_P / 2, mf, vf)
    k3_m, k3_P = m_cov_ode(m + dt * k2_m / 2, v + dt * k2_P / 2, mf, vf)
    k4_m, k4_P = m_cov_ode(m + dt * k3_m, v + dt * k3_P, mf, vf)
    return m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6, \
           v + dt * (k1_P + 2 * k2_P + 2 * k3_P + k4_P) / 6


class SigmaPoints(NamedTuple):
    r"""Sigma-point integration.

    .. math::

        \int z(x) \mathrm{N}(x \mid m, P) dx \approx \sum^s_{i=1} z(\chi_i),

    where :math:`\chi_i = m + \sqrt{P} \, \xi_i`.

    Attributes
    ----------
    d : int
        Problem dimension (e.g., state dimension).
    n_points : int
        Number of sigma points.
    w : JArray (s, )
        Weights.
    wc : JArray (sc, )
        Additional weights (if has any).
    xi : JArray (s, d)
        Pre-sigma points.
    """
    d: int
    n_points: int
    w: JArray
    wc: Union[JArray, None]
    xi: JArray

    @staticmethod
    def _gen_hermite_poly_coeff(order: int) -> List[np.ndarray]:
        """Give the 0 to p-th order physician Hermite polynomial coefficients, where p is the
        order argument. The returned coefficients is ordered from highest to lowest.
        Also note that this implementation is different from the np.hermite method.

        Parameters
        ----------
        order : int
            The order of Hermite polynomial

        Returns
        -------
        H : List
            The 0 to p-th order Hermite polynomial coefficients in a list.
        """
        H0 = np.array([1])
        H1 = np.array([2, 0])

        H = [H0, H1]

        for i in range(2, order + 1):
            H.append(2 * np.append(H[i - 1], 0) -
                     2 * (i - 1) * np.pad(H[i - 2], (2, 0), 'constant', constant_values=0))
        return H

    @classmethod
    def cubature(cls, d: int):
        """A factory method for generating spherical cubature :code:`SigmaPoints`.

        Parameters
        ----------
        d : int
            State dimension.
        """
        n_points = 2 * d
        w = jnp.ones(shape=(n_points,)) / n_points
        xi = math.sqrt(d) * jnp.concatenate([jnp.eye(d), -jnp.eye(d)], axis=0)
        return cls(d=d, n_points=n_points, w=w, wc=None, xi=xi)

    @classmethod
    def unscented(cls, d: int, alpha: float, beta: float, lam: float):
        raise NotImplementedError('Unscented transform is not implemented.')

    @classmethod
    def gauss_hermite(cls, d: int, order: int = 3):
        """A factory method for generating Gauss--Hermite :code:`SigmaPoints`.

        Parameters
        ----------
        d : int
            State dimension.
        order : int, default=3
            Order of Hermite polynomial.
        """
        n_points = order ** d

        hermite_coeff = cls._gen_hermite_poly_coeff(order)
        hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

        table = np.zeros(shape=(d, order ** d))

        w_1d = np.zeros(shape=(order,))
        for i in range(order):
            w_1d[i] = (2 ** (order - 1) * np.math.factorial(order) * np.sqrt(np.pi) /
                       (order ** 2 * (np.polyval(hermite_coeff[order - 1],
                                                 hermite_roots[i])) ** 2))

        # Get roll table
        for i in range(d):
            base = np.ones(shape=(1, order ** (d - i - 1)))
            for j in range(1, order):
                base = np.concatenate([base,
                                       (j + 1) * np.ones(shape=(1, order ** (d - i - 1)))],
                                      axis=1)
            table[d - i - 1, :] = np.tile(base, (1, int(order ** i)))

        table = table.astype("int64") - 1

        s = 1 / (np.sqrt(np.pi) ** d)

        w = s * np.prod(w_1d[table], axis=0)
        xi = math.sqrt(2) * hermite_roots[table].T

        return cls(d=d, n_points=n_points, w=jnp.array(w), wc=None, xi=jnp.array(xi))

    def gen_sigma_points(self, m: JArray, chol_of_v: JArray) -> JArray:
        r"""Generate sigma points :math:`\lbrace \chi_i = m + \sqrt{P} xi_i \rbrace^s_{i=1}`.
        """
        return m + jnp.einsum('ij,...j->...i', chol_of_v, self.xi)

    def expectation_from_nodes(self, v_f: Callable, chi: JArray) -> JArray:
        r"""Approximate expectation by using sigma points.

        Parameters
        ----------
        v_f : Callable (s, ...) -> (s, ???)
            Vectorised integrand function.
        chi : JArray (s, ...)

        Returns
        -------
        JArray (...)
        """
        return jnp.einsum('i,i...->...', self.w, v_f(chi))

    def expectation(self, evals_of_integrand: JArray) -> JArray:
        """Approximate expectation by using sigma points.

        The same as with :code:`expectation_from_nodes`, except feeding the propagated integration nodes.

        Parameters
        ----------
        evals_of_integrand : (s, ...)

        Returns
        -------
        JArray (...)
        """
        return jnp.einsum('i,i...->...', self.w, evals_of_integrand)


def gaussian_expectation(ms: JArray, chol_vs: JArray, func: Callable,
                         d: int = 1, order: int = 10, force_shape: bool = False):
    r"""Approximate :math:`E[g(V_{1:T})]` with Gauss--Hermite, where random variables V \sim N(m, P).

    Parameters
    ----------
    ms : JArray (T, d)
        T number of means.
    chol_vs : JArray (T, d, d)
        T number of Cholesky of covariances.
    func : Callable
        The function that we take expectation on.
    d : int, default=1
        Dimension of V.
    order : int, default=10
        Gauss--Hermite integration order.
    force_shape : bool, default=False
        Force to reshape ms and chol_Ps to (-1, 1) and (-1, 1, 1), respectively.

    Returns
    -------
    JArray (T, d)
        E[g(V)].

    Notes
    -----
    In this chirp application the dimension of V is 1.
    """
    # Reshape ms and chol_Ps if possible
    if force_shape:
        ms = jnp.reshape(ms, (-1, 1))
        chol_vs = jnp.reshape(chol_vs, (-1, 1, 1))

    sgps = SigmaPoints.gauss_hermite(d=d, order=order)

    @partial(jax.vmap, in_axes=[0, 0])
    def approximate(m, chol):
        chi = sgps.gen_sigma_points(m, chol)
        return sgps.expectation_from_nodes(func, chi)

    return approximate(ms, chol_vs)
