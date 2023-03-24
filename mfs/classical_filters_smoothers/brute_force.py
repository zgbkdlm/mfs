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
Solve the 1D filtering problem by brute-force computing the PDFs.
"""
import jax
import jax.numpy as jnp
import tme.base_jax as tme
from mfs.typings import JArray, FloatScalar
from typing import Callable


def brute_force_filter(drift: Callable, dispersion: Callable, measurement_cond_pdf: Callable,
                       init_ps: JArray, xs: JArray, ys: JArray, dt: FloatScalar,
                       integration_steps: int = 1,
                       pred_method: str = 'chapman-tme-2') -> JArray:
    """Brute-force computing the true filtering solution (for 1D state only).

    Parameters
    ----------
    drift : Callable
        The SDE drift function.
    dispersion : Callable
        The SDE dispersion function.
    measurement_cond_pdf : Callable ..., (n, ) -> (n, )
        The conditional PDF of the measurement variable.
    init_ps : JArray (n, )
        The initial probability density values evaluated at `xs`.
    xs : JArray (n, )
        The spatial location where the PDFs are evaluated. Note that this implementation only supports evenly
        partitioned grids, e.g., `jnp.linspace(...)`, if you are using the kolmogorov solver.
    ys : JArray (T, ...)
        Measurements.
    dt : FloatScalar
        The time interval between two measurements.
    integration_steps : int, default=1
        The number of integration steps between two measurement times.
    pred_method : str, default='chapman-tme-2'
        The method that computes the predictive density. Implemented methods are:

            * 'kolmogorov'. Solve the Kolmogorov forward equation by finite difference and Euler propagation.
            * 'chapman-euler'. Approximate the transition density by Euler--Maruyama then trapezoidal integration.
            * 'chapman-tme-?'. Approximate the transition density by TME of order `?` then trapezoidal integration.

        When using `kolmogorov`, be careful with the numerical stability. Usually `chapman-tme-2` is already far better
        than `chapman-euler` at the cost of slightly more computations.

    Returns
    -------
    JArray (T, n)
        The filtering PDFs evaluated at the temporal and spatial locations.
    """
    dx = xs[1] - xs[0]
    ddt = dt / integration_steps

    if pred_method == 'chapman-euler':
        m, scale = xs + drift(xs) * ddt, dispersion(xs) * jnp.sqrt(ddt)
    elif 'chapman-tme' in pred_method:
        order = int(pred_method.split('-')[-1])

        def _m_scale(x):
            _m, _cov = tme.mean_and_cov(x, ddt, drift, dispersion, order=order)
            return jnp.squeeze(_m), jnp.squeeze(jnp.sqrt(_cov))

        m, scale = jax.vmap(_m_scale, in_axes=[0])(xs[:, None])

    def gamma(x):
        return dispersion(x) ** 2

    def integrate_transition(x, ps):
        r"""\int p(x_k | x_{k-1}) p(x_{k-1}) dx_{k-1}
        """
        return jnp.trapz(jax.scipy.stats.norm.pdf(x, m, scale) * ps, xs)

    derivative_drift = jax.vmap(jax.grad(drift), in_axes=[0])
    derivative_gamma = jax.vmap(jax.grad(gamma), in_axes=[0])
    second_derivative_gamma = jax.vmap(jax.grad(jax.grad(gamma)), in_axes=[0])

    def kolmogorov_fwd_opt(ps):
        """Kolmogorov forward operator.
        """
        derivative_ps = jnp.gradient(ps, dx)
        second_derivative_ps = jnp.gradient(jnp.gradient(ps, dx), dx)

        part1 = -(derivative_drift(xs) * ps + drift(xs) * derivative_ps)
        return part1 + 0.5 * (
                second_derivative_gamma(xs) * ps
                + 2 * derivative_gamma(xs) * derivative_ps + gamma(xs) * second_derivative_ps)

    def euler(ps):
        return ps + kolmogorov_fwd_opt(ps) * ddt

    def pred_kolmogorov(ps):
        def body(carry, _):
            _ps = carry
            _ps = euler(_ps)
            return _ps, None

        return jax.lax.scan(body, ps, jnp.arange(integration_steps))[0]

    def pred_chapman(ps):
        def body(carry, _):
            _ps = carry
            _ps = jax.vmap(integrate_transition, in_axes=[0, None])(xs, _ps)
            return _ps, None

        return jax.lax.scan(body, ps, jnp.arange(integration_steps))[0]

    def scan_body(carry, elem):
        ps = carry
        y = elem

        if pred_method == 'kolmogorov':
            ps = pred_kolmogorov(ps)
        elif 'chapman' in pred_method:
            ps = pred_chapman(ps)
        else:
            raise NotImplementedError(f'Prediction method {pred_method} not implemented.')

        ps = measurement_cond_pdf(y, xs) * ps / jnp.trapz(measurement_cond_pdf(y, xs) * ps, xs)
        return ps, ps

    return jax.lax.scan(scan_body, init_ps, ys)[1]
