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
A few implementations of commonly used stochastic filters and smoothers in discrete time and continuous-discrete time.
"""
import math
import jax
import jax.numpy as jnp
import jax.scipy
from .quadratures import SigmaPoints, rk4_m_cov_backward, rk4_m_cov
from mfs.typings import JArray, FloatScalar, JFloat
from typing import Callable, Tuple
from functools import partial

__all__ = ['kf',
           'rts',
           'ekf',
           'eks',
           'cd_ekf',
           'cd_eks',
           'sgp_filter',
           'sgp_smoother',
           'cd_sgp_filter',
           'cd_sgp_smoother']


@partial(jax.vmap, in_axes=[0, 0])
def _vectorised_outer(x: JArray, y: JArray) -> JArray:
    return jnp.outer(x, y)


def _log_normal_pdf(x: FloatScalar, mu: FloatScalar, variance: FloatScalar) -> FloatScalar:
    return jax.scipy.stats.norm.logpdf(x, mu, jnp.sqrt(variance))


def _log_mvn_pdf(x: JArray, mu: JArray, chol: JArray) -> JFloat:
    z = jax.lax.linalg.triangular_solve(chol, x - mu)
    return -0.5 * (jnp.dot(z, z) + 2 * jnp.sum(jnp.log(math.sqrt(2 * math.pi) * jnp.diagonal(chol))))


def _linear_predict(F: JArray, Sigma: JArray,
                    m: JArray, P: JArray) -> Tuple[JArray, JArray]:
    """Mean and covariance of X_k from X_k = F X_{k-1} + Q_{k-1}.
    """
    return F @ m, F @ P @ F.transpose() + Sigma


def _linear_update(mp: JArray, vp: JArray,
                   H: JArray, pred_y: JArray, Xi: JArray, y: JArray) -> Tuple[JArray, JArray, JFloat]:
    """Update for linear Gaussian measurement models
    (note that here the dimension of the measurement variable is assumed to be 1).

    Returns
    -------
    JArray, JArray, DeviceFloat
        Mean, covariance, and negative log likelihood.
    """
    S = H @ vp @ H.T + Xi

    if H.shape[0] == 1:
        K = vp @ H.T / S
        n_ell = jnp.squeeze(-_log_normal_pdf(y, pred_y, S))
    else:
        chol = jax.lax.linalg.cholesky(S)
        K = jax.lax.linalg.triangular_solve(chol, H @ vp).T
        n_ell = -_log_mvn_pdf(y, pred_y, chol)
    return mp + K @ (y - pred_y), vp - K @ S @ K.T, n_ell


def _gaussian_smoother_shared(DT: JArray,
                              mf: JArray, vf: JArray,
                              mp: JArray, vp: JArray,
                              ms: JArray, vs: JArray) -> Tuple[JArray, JArray]:
    """Shared procedure for Gaussian smoothers as per Equation 2.25 in Zhao 2021.

    Notes
    -----
    DT is the transpose of D.
    """
    c, low = jax.scipy.linalg.cho_factor(vp)
    G = jax.scipy.linalg.cho_solve((c, low), DT).T
    ms = mf + G @ (ms - mp)
    vs = vf + G @ (vs - vp) @ G.T
    return ms, vs


def _sgp_prediction(sgps: SigmaPoints,
                    vectorised_state_cond_m_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
                    dt: FloatScalar,
                    mf: JArray,
                    vf: JArray) -> Tuple[JArray, JArray, JArray, JArray]:
    r"""Sigma-point prediction of state-space model.

    Return the sigma-point approximated mean and covariance of :math:`X_k` from a state-space defined by the
    conditional mean and covariance :code:`cond_m_cov` starting from :math:`X_{k-1} \sim N(mf, vf)`.

    Parameters
    ----------
    sgps : SigmaPoints
        Sigma-points object.
    vectorised_state_cond_m_cov : Callable (..., dx), () -> (..., dx), (..., dx, dx)
        A function that returns the conditional mean and covariance. Furthermore, the function vectorises over the first
        argument.
    dt : FloatScalar
        Time interval between t_k and t_{k-1}.
    mf : JArray (dx,)
        Initial mean of this prediction.
    vf : JArray (dx, dx)
        Initial covariance of this prediction.

    Returns
    -------
    JArray (dx,), JArray (dx, dx), JArray (..., dx), JArray (..., dx)
        Sigma-points predicted mean and covariance. The last two returns are used by the sgp smoother.
    """
    chol_vf = jax.lax.linalg.cholesky(vf)
    chi = sgps.gen_sigma_points(mf, chol_vf)

    evals_of_m, evals_of_cov = vectorised_state_cond_m_cov(chi, dt)
    mp = sgps.expectation(evals_of_m)
    vp = sgps.expectation(_vectorised_outer(evals_of_m, evals_of_m) + evals_of_cov) - jnp.outer(mp, mp)
    return mp, vp, chi, evals_of_m


def _sgp_update(sgps: SigmaPoints,
                vectorised_measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
                mp: JArray,
                vp: JArray,
                y: JArray,
                const_measurement_cov: bool = False) -> Tuple[JArray, JArray, JFloat]:
    r"""Sigma-point update of state-space model.

    Return the sigma-point approximated mean and covariance of :math:`X_k` from a state-space defined by the
    conditional mean and covariance :code:`cond_m_cov` starting from :math:`X_{k-1} \sim N(mf, vf)`.

    Parameters
    ----------
    sgps : SigmaPoints
        Sigma-points object.
    vectorised_measurement_cond_m_cov : Callable (..., dx), -> (..., dy), (..., dy, dy)
        The measurement function that returns vectorised mean and covariance.
    mp : JArray (dx,)
        Initial mean of this prediction.
    vp : JArray (dx, dx)
        Initial covariance of this prediction.
    y : JArray (dy,)
        Measurement.
    const_measurement_cov : bool, default=False
        Whether the measurement noise covariance is constant (which is the second return of
        :code:`vectorised_measurement_func`).

    Returns
    -------
    JArray (dx,), JArray (dx, dx), DeviceFloat
        Sigma-points integrated filtering mean, covariance, and negative log likelihood.
    """
    chol_vp = jax.lax.linalg.cholesky(vp)
    chi = sgps.gen_sigma_points(mp, chol_vp)

    evals_m, evals_Xi = vectorised_measurement_cond_m_cov(chi)
    pred = sgps.expectation(evals_m)
    if const_measurement_cov:
        S = sgps.expectation(_vectorised_outer(evals_m, evals_m)) - jnp.outer(pred, pred) + evals_Xi[0]
    else:
        S = sgps.expectation(_vectorised_outer(evals_m, evals_m) + evals_Xi) - jnp.outer(pred, pred)
    C = sgps.expectation(_vectorised_outer(chi, evals_m)) - jnp.outer(mp, sgps.expectation(evals_m))

    chol = jax.lax.linalg.cholesky(S)
    K = jax.scipy.linalg.cho_solve((chol, True), C.T).T
    return mp + K @ (y - pred), vp - K @ S @ K.T, -_log_mvn_pdf(y, pred, chol)


def _ekf_update(measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
                mp: JArray,
                vp: JArray,
                y: JArray,
                fwd_jacobian: bool = False) -> Tuple[JArray, JArray, JFloat]:
    r"""Update by Taylor expansion.

    Parameters
    ----------
    measurement_cond_m_cov : Callable (dx, ) -> (dy, ), (dy, dy)
        ...
    mp : JArray (dx, )
        Initial mean of this prediction.
    vp : JArray (dx, dx)
        Initial covariance of this prediction.
    y : JArray (dy, )
        Measurement.
    fwd_jacobian : bool, defaultFalse
        Mode of Jacobian. Use forward when dy >> dx, otherwise use jacrev.

    Returns
    -------
    JArray (dx,), JArray (dx, dx), DeviceFloat
        Sigma-points integrated filtering mean, covariance, and negative log likelihood.
    """
    if fwd_jacobian:
        H = jax.jacfwd(measurement_cond_m_cov)(mp)[0]
    else:
        H = jax.jacrev(measurement_cond_m_cov)(mp)[0]
    pred_y_m, pred_y_cov = measurement_cond_m_cov(mp)
    return _linear_update(mp, vp, H, pred_y_m, pred_y_cov, y)


def _cd_sgp_shared(sgps: SigmaPoints,
                   vectorised_drift: Callable[[JArray], JArray],
                   dispersion_const: JArray,
                   m: JArray, P: JArray) -> Tuple[JArray, JArray]:
    r"""Sigma-point prediction of SDE model.
    """
    chol_v = jax.lax.linalg.cholesky(P)
    chi = sgps.gen_sigma_points(m, chol_v)

    evals_of_drift = vectorised_drift(chi)
    mp = sgps.expectation(evals_of_drift)
    _vp = sgps.expectation(_vectorised_outer(chi - m, evals_of_drift))
    vp = _vp + _vp.T + dispersion_const @ dispersion_const.T
    return mp, vp


def _stack_smoothing_results(mfs: JArray, vfs: JArray,
                             mss: JArray, vss: JArray) -> Tuple[JArray, JArray]:
    return jnp.vstack([mss, mfs[-1]]), jnp.vstack([vss, vfs[-1, None]])


def kf(F: JArray, Sigma: JArray,
       H: JArray, Xi: JArray,
       m0: JArray, v0: JArray,
       ys: JArray) -> Tuple[JArray, JArray, JArray]:
    """Kalman filter.

    Parameters
    ----------
    F : JArray (dx, dx)
        State transition mean matrix.
    Sigma : JArray (dx, dx)
        State transition covariance.
    H : JArray (dy, dx)
        Measurement matrix.
    Xi : JArray (dy, dy)
        Measurement variance.
    m0 : JArray (dx, )
        Initial mean.
    v0 : JArray (dx, dx)
        Initial covariance.
    ys : JArray (T, dy)
        Measurements.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx), JArray (T, )
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    def scan_body(carry, elem):
        mf, vf, n_ell = carry
        y = elem

        mp, vp = _linear_predict(F, Sigma, mf, vf)
        mf, vf, n_ell_inc = _linear_update(mp, vp, H, H @ mp, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, vf, n_ell), (mf, vf, n_ell)

    _, (mfs, vfs, n_ell) = jax.lax.scan(scan_body, (m0, v0, 0.), ys)
    return mfs, vfs, n_ell


def rts(F: JArray, Sigma: JArray,
        mfs: JArray, vfs: JArray) -> Tuple[JArray, JArray]:
    """RTS smoother.

    Parameters
    ----------
    F : JArray (dx, dx)
        State transition mean matrix.
    Sigma : JArray (dx, dx)
        State transition covariance.
    mfs : JArray (T, dx)
        Filtering posterior means.
    vfs : JArray (T, dx, dx)
        Filtering posterior covariances.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx)
        Means and covariances of the smoothing estimates.
    """

    def scan_body(carry, elem):
        ms, vs = carry
        mf, vf = elem

        ms, vs = _gaussian_smoother_shared(F @ vf,
                                           mf, vf,
                                           F @ mf, F @ vf @ F.T + Sigma,
                                           ms, vs)
        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_body, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, vfs, mss, vss)


def ekf(state_cond_m_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
        measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
        m0: JArray,
        v0: JArray,
        dt: FloatScalar,
        ys: JArray,
        fwd_jacobian: bool = False) -> Tuple[JArray, JArray, JArray]:
    """Extended Kalman filter for non-linear models.

    Parameters
    ----------
    state_cond_m_cov : Callable ((dx, ), FloatScalar) -> (d, ), (d, d)
        A function that returns the conditional mean and covariance of SDE.
    measurement_cond_m_cov : Callable (dx, ) -> (dy, ), (dy, dy)
        Measurement function.
    m0 : JArray (dx, )
        Initial mean.
    v0 : JArray (dx, dx)
        Initial covariance.
    dt : FloatScalar
        Time interval.
    ys : JArray (T, dy)
        Measurements.
    fwd_jacobian : bool, defaultFalse
        Mode of Jacobian. Use forward when dy >> dx, otherwise use jacrev.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx), JArray (T, )
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    def scan_body(carry, elem):
        mf, vf, n_ell = carry
        y = elem

        jac_F = jax.jacfwd(lambda u: state_cond_m_cov(u, dt)[0], argnums=0)(mf)
        mp, Sigma = state_cond_m_cov(mf, dt)
        vp = jac_F @ vf @ jac_F.T + Sigma

        mf, vf, n_ell_inc = _ekf_update(measurement_cond_m_cov, mp, vp, y, fwd_jacobian)
        n_ell = n_ell + n_ell_inc
        return (mf, vf, n_ell), (mf, vf, n_ell)

    _, (mfs, vfs, n_ell) = jax.lax.scan(scan_body, (m0, v0, 0.), ys)
    return mfs, vfs, n_ell


def eks(state_cond_m_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
        mfs: JArray,
        vfs: JArray,
        dt: FloatScalar) -> Tuple[JArray, JArray]:
    """Extended Kalman smoother for non-linear dynamical models.

    Parameters
    ----------
    state_cond_m_cov : Callable ((dx, ), FloatScalar) -> (dx, ), (dx, dx)
        A function that returns the conditional mean and covariance of SDE.
    mfs : JArray (T, dx)
        Filtering posterior means.
    vfs : JArray (T, dx, dx)
        Filtering posterior covariances.
    dt : FloatScalar
        Time interval.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx)
        Means and covariances of the smoothing estimates.
    """

    def scan_body(carry, elem):
        ms, vs = carry
        mf, vf = elem

        jac_F = jax.jacfwd(lambda u: state_cond_m_cov(u, dt)[0], argnums=0)(mf)
        mp, Sigma = state_cond_m_cov(mf, dt)
        vp = jac_F @ vf @ jac_F.T + Sigma
        ms, vs = _gaussian_smoother_shared(jac_F @ vf, mf, vf, mp, vp, ms, vs)
        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_body, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, vfs, mss, vss)


def cd_ekf(drift: Callable[[JArray], JArray],
           dispersion: Callable[[JArray], JArray],
           measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
           m0: JArray,
           v0: JArray,
           dt: FloatScalar,
           ys: JArray,
           fwd_jacobian: bool = False) -> Tuple[JArray, JArray, JArray]:
    """Continuous-discrete extended Kalman filter with 4th order Runge--Kutta integration.

    Parameters
    ----------
    drift : Callable (dx, ) -> (dx, )
        SDE drift function.
    dispersion : Callable (dx, dw) -> (dx, dw)
        SDE dispersion function.
    measurement_cond_m_cov : (dx, ) -> (dy, ), (dy, dy)
        Measurement function.
    Xi : FloatScalar
        Measurement noise cvariance.
    m0 : JArray (dx, )
        Initial mean.
    v0 : JArray (dx, dx)
        Initial covariance.
    dt : float
        Time interval
    ys : JArray (T, dy)
        Measurements.
    fwd_jacobian : bool, defaultFalse
        Mode of Jacobian. Use forward when dy >> dx, otherwise use jacrev.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx), JArray (T, )
        Filtering posterior means and covariances, and negative log likelihoods.
    """
    jac_of_drift = jax.jacfwd(drift)

    def odes(m, v):
        return drift(m), v @ jac_of_drift(m).T + jac_of_drift(m) @ v + dispersion(m) @ dispersion(m).T

    def scan_body(carry, elem):
        mf, vf, n_ell = carry
        y = elem

        mp, vp = rk4_m_cov(odes, mf, vf, dt)
        mf, vf, n_ell_inc = _ekf_update(measurement_cond_m_cov, mp, vp, y, fwd_jacobian)
        n_ell = n_ell + n_ell_inc
        return (mf, vf, n_ell), (mf, vf, n_ell)

    _, filtering_results = jax.lax.scan(scan_body, (m0, v0, 0.), ys)
    return filtering_results


def cd_eks(drift: Callable[[JArray], JArray],
           dispersion: Callable[[JArray], JArray],
           mfs: JArray, vfs: JArray,
           dt: FloatScalar) -> Tuple[JArray, JArray]:
    """Continuous-discrete extended Kalman smoother with 4th order Runge--Kutta integration.

    Parameters
    ----------
    drift : Callable (dx, ) -> (dx, )
        SDE drift function.
    dispersion : Callable (dx, dw) -> (dx, dw)
        SDE dispersion function.
    mfs : JArray (T, dx)
        Filtering means.
    vfs : JArray (T, dx, dx)
        Filtering covariances.
    dt : FloatScalar
        Time interval

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx)
        Mean and covariance of the smoothing estimates.
    """
    dt = -dt

    jac_of_drift = jax.jacfwd(drift)

    def odes(m, v, mf, vf):
        gamma = dispersion(m) @ dispersion(m).T
        c, low = jax.scipy.linalg.cho_factor(vf)
        jac_and_gamma_and_chol = jac_of_drift(m) + jax.scipy.linalg.cho_solve((c, low), gamma.T).T
        return drift(m) + gamma @ jax.scipy.linalg.cho_solve((c, low), m - mf), \
               jac_and_gamma_and_chol @ v + v @ jac_and_gamma_and_chol.T - gamma

    def scan_body(carry, elem):
        ms, vs = carry
        mf, vf = elem

        ms, vs = rk4_m_cov_backward(odes, ms, vs, mf, vf, dt)

        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_body, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, vfs, mss, vss)


def sgp_filter(state_cond_m_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
               measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
               sgps: SigmaPoints,
               m0: JArray,
               v0: JArray,
               dt: FloatScalar,
               ys: JArray, const_measurement_cov: bool = False) -> Tuple[JArray, JArray, JArray]:
    """Continuous-discrete sigma-point filter by discretising the SDE.

    Parameters
    ----------
    state_cond_m_cov : Callable (dx, ), FloatScalar -> (dx, ), (dx, dx)
        A function that returns the conditional mean and covariance of SDE.
    measurement_cond_m_cov : (dx, ) -> (dy, ), (dy, dy)
        Measurement function.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    m0 : JArray (dx, )
        Initial mean.
    v0 : JArray (dx, dx)
        Initial covariance.
    dt : FloatScalar
        Time interval.
    ys : JArray (T, dy)
        Measurements.
    const_measurement_cov : bool, default=False
        Whether the measurement noise covariance is constant (which is the second return of
        :code:`vectorised_measurement_func`).

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx), JArray (T, )
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    vectorised_cond_m_cov = jax.vmap(state_cond_m_cov, in_axes=[0, None])
    vectorised_measurement_func = jax.vmap(measurement_cond_m_cov, in_axes=[0])

    def scan_sgp_filter(carry, elem):
        mf, vf, n_ell = carry
        y = elem

        mp, vp, _, _ = _sgp_prediction(sgps, vectorised_cond_m_cov, dt, mf, vf)
        mf, vf, n_ell_inc = _sgp_update(sgps, vectorised_measurement_func, mp, vp, y, const_measurement_cov)
        n_ell = n_ell + n_ell_inc
        return (mf, vf, n_ell), (mf, vf, n_ell)

    _, (mfs, vfs, n_ell) = jax.lax.scan(scan_sgp_filter, (m0, v0, 0.), ys)
    return mfs, vfs, n_ell


def sgp_smoother(state_cond_m_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
                 sgps: SigmaPoints,
                 mfs: JArray,
                 vfs: JArray,
                 dt: FloatScalar) -> Tuple[JArray, JArray]:
    """Continuous-discrete sigma-point smoother by discretising the SDE.

    Parameters
    ----------
    state_cond_m_cov : (dx, ), FloatScalar -> (dx, ), (dx, dx)
        A function that returns the conditional mean and covariance of SDE.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    mfs : JArray (T, dx)
        Filtering means.
    vfs : JArray (T, dx, dx)
        Filtering covariances.
    dt : FloatScalar
        Time interval.

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx)
        Means and covariances of the smoothing estimates.
    """

    vectorised_cond_m_cov = jax.vmap(state_cond_m_cov, in_axes=[0, None])

    def scan_sgp_smoother(carry, elem):
        ms, vs = carry
        mf, vf = elem

        mp, vp, chi, evals_of_m = _sgp_prediction(sgps, vectorised_cond_m_cov, dt, mf, vf)
        D = sgps.expectation(_vectorised_outer(chi, evals_of_m)) - jnp.outer(mf, mp)

        ms, vs = _gaussian_smoother_shared(D.T, mf, vf, mp, vp, ms, vs)
        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_sgp_smoother, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, vfs, mss, vss)


def cd_sgp_filter(drift: Callable[[JArray], JArray],
                  dispersion: JArray,
                  measurement_cond_m_cov: Callable[[JArray], Tuple[JArray, JArray]],
                  sgps: SigmaPoints,
                  m0: JArray,
                  v0: JArray,
                  dt: FloatScalar,
                  ys: JArray,
                  const_measurement_cov: bool = False) -> Tuple[JArray, JArray, JArray]:
    """Continuous-discrete sigma-points Kalman filter with 4th order Runge--Kutta integration.

    Parameters
    ----------
    drift : Callable (dx, ) -> (dx, )
        SDE drift function.
    dispersion : JArray (d, dw)
        SDE dispersion matrix.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    measurement_cond_m_cov: (dx, ) -> (dy, ), (dy, dy)
        Measurement function.
    m0 : JArray (dx, )
        Initial mean.
    v0 : JArray (dx, dx)
        Initial covariance.
    dt : FloatScalar
        Time interval
    ys : JArray (T, dx)
        Measurements.
    const_measurement_cov : bool, default=False
        Whether the measurement noise covariance is constant (which is the second return of
        :code:`vectorised_measurement_func`).

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx), JArray (T, )
        Filtering posterior means and covariances, and negative log likelihoods.
    """
    vectorised_drift = jax.vmap(drift, in_axes=[0])
    vectorised_measurement_func = jax.vmap(measurement_cond_m_cov, in_axes=[0])

    def odes(m, v):
        return _cd_sgp_shared(sgps, vectorised_drift, dispersion, m, v)

    def scan_body(carry, elem):
        mf, vf, n_ell = carry
        y = elem

        mp, vp = rk4_m_cov(odes, mf, vf, dt)
        mf, vf, n_ell_inc = _sgp_update(sgps, vectorised_measurement_func, mp, vp, y, const_measurement_cov)
        n_ell = n_ell + n_ell_inc
        return (mf, vf, n_ell), (mf, vf, n_ell)

    _, filtering_results = jax.lax.scan(scan_body, (m0, v0, 0.), ys)
    return filtering_results


def cd_sgp_smoother(drift: Callable[[JArray], JArray],
                    dispersion: JArray,
                    sgps: SigmaPoints,
                    mfs: JArray,
                    vfs: JArray,
                    dt: FloatScalar) -> Tuple[JArray, JArray]:
    """Continuous-discrete sigma-points Kalman smoother with 4th order Runge--Kutta integration.

    Parameters
    ----------
    drift : Callable (dx, ) -> (dx, )
        SDE drift function.
    dispersion : JArray (dx, dw)
        SDE dispersion matrix.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    mfs : JArray (T, dx)
        Filtering means.
    vfs : JArray (T, dx, dx)
        Filtering covariances.
    dt : FloatScalar
        Time interval

    Returns
    -------
    JArray (T, dx), JArray (T, dx, dx)
        Mean and covariance of the smoothing estimates.
    """
    dt = -dt

    vectorised_drift = jax.vmap(drift, in_axes=[0])

    def odes(m, v, mf, vf):
        gamma = dispersion @ dispersion.T
        c, low = jax.scipy.linalg.cho_factor(vf)
        G = jax.scipy.linalg.cho_solve((c, low), gamma)

        _m, _P = _cd_sgp_shared(sgps, vectorised_drift, dispersion, m, v)
        return _m + G.T @ (m - mf), _P + G.T @ v + v @ G - 2 * gamma

    def scan_body(carry, elem):
        ms, vs = carry
        mf, vf = elem

        ms, vs = rk4_m_cov_backward(odes, ms, vs, mf, vf, dt)

        return (ms, vs), (ms, vs)

    _, (mss, vss) = jax.lax.scan(scan_body, (mfs[-1], vfs[-1]), (mfs[:-1], vfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, vfs, mss, vss)
