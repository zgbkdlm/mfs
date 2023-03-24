"""
Test baseline filters.
"""
import pytest
import math
import jax
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from mfs.classical_filters_smoothers import kf, rts, ekf, eks, cd_ekf, cd_eks, \
    sgp_filter, sgp_smoother, cd_sgp_filter, cd_sgp_smoother
from mfs.classical_filters_smoothers.smc import bootstrap_filter, particle_filter
from mfs.classical_filters_smoothers.resampling import stratified
from mfs.classical_filters_smoothers.brute_force import brute_force_filter
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from mfs.one_dim.moments import characteristic_from_pdf
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)


class TestFiltersSmoothers:

    @pytest.mark.parametrize('flag', [True, False])
    @pytest.mark.parametrize('a, b', ([1., 1.],
                                      [2.1, 0.4]))
    def test_equivalence_on_linear_models(self, a, b, flag):

        dim_x = 3
        dt = 0.01

        # dx = A x dt + B dW
        A = -a * jnp.eye(dim_x)
        B = b * jnp.eye(dim_x)

        drift = lambda u: A @ u
        dispersion = lambda _: B

        # x_k = F x_{k-1} + Q
        F = math.exp(-a * dt) * jnp.eye(dim_x)
        Sigma = b ** 2 / (2 * a) * (1 - math.exp(-2 * a * dt)) * jnp.eye(dim_x)

        Xi = jnp.array([[0.1]])
        H = jnp.ones((1, dim_x))

        m0 = jnp.zeros((dim_x,))
        P0 = 0.1 * jnp.eye(dim_x)

        def simulate():
            num_measurements = 100
            xx = np.zeros((num_measurements, dim_x))
            yy = np.zeros((num_measurements, 1))
            x = np.array(m0).copy()
            for i in range(num_measurements):
                x = F @ x + np.sqrt(Sigma) @ np.random.randn(dim_x)
                y = H @ x + np.sqrt(Xi) * np.random.randn()
                xx[i] = x
                yy[i, :] = y

            return jnp.asarray(xx), jnp.asarray(yy)

        def m_and_cov(u, _):
            return F @ u, Sigma

        def measurement_func(u):
            return H @ u, Xi

        xs, ys = simulate()

        kf_results = kf(F, Sigma, H, Xi, m0, P0, ys)
        ekf_results = ekf(m_and_cov, measurement_func, m0, P0, dt, ys, fwd_jacobian=flag)
        cd_ekf_results = cd_ekf(drift, dispersion, measurement_func, m0, P0, dt, ys, fwd_jacobian=flag)
        sgps = SigmaPoints.gauss_hermite(d=dim_x, order=4)
        ghkf_results = sgp_filter(m_and_cov, measurement_func, sgps, m0, P0, dt, ys, const_measurement_cov=flag)
        cd_ghkf_results = cd_sgp_filter(drift, B, measurement_func, sgps, m0, P0, dt, ys, const_measurement_cov=flag)

        # Particle filters
        def transition_density(x, ancestors):
            return jax.scipy.stats.multivariate_normal.pdf(x, ancestors @ F, Sigma)

        def proposal_sampler(ancestors, _, _key):
            m, cov = ancestors @ F, Sigma
            return m + jax.random.normal(_key, ancestors.shape) @ jnp.sqrt(cov)

        def proposal_density(x, ancestors, _):
            m, cov = ancestors @ F, Sigma
            return jax.scipy.stats.multivariate_normal.pdf(x, m, cov)

        def init_sampler(_key, n: int):
            return m0 + jax.random.normal(_key, (n, dim_x)) @ jnp.sqrt(P0)

        def measurement_cond_pdf(y, x):
            return jax.scipy.stats.norm.pdf(y, x @ H[0], math.sqrt(Xi))

        pf_results = particle_filter(proposal_sampler, proposal_density, transition_density,
                                     measurement_cond_pdf, ys[:, 0],
                                     init_sampler, jax.random.PRNGKey(999), 10000, stratified)
        bf_results = bootstrap_filter(lambda u, v: proposal_sampler(u, None, v),
                                      measurement_cond_pdf, ys, init_sampler, jax.random.PRNGKey(999),
                                      10000, stratified)[0]

        # Test filtering results
        for i in range(3):
            npt.assert_allclose(kf_results[i], ekf_results[i])
            npt.assert_allclose(kf_results[i], ghkf_results[i])
            npt.assert_allclose(kf_results[i], cd_ekf_results[i], rtol=1e-5)
            npt.assert_allclose(kf_results[i], cd_ghkf_results[i], rtol=1e-5)

        npt.assert_array_equal(pf_results, bf_results)
        npt.assert_allclose(kf_results[0], np.mean(pf_results, 1), atol=2e-1)

        # Test smoothing results
        rts_results = rts(F, Sigma, kf_results[0], kf_results[1])
        eks_results = eks(m_and_cov, ekf_results[0], ekf_results[1], dt)
        cd_eks_results = cd_eks(drift, dispersion, cd_ekf_results[0], cd_ekf_results[1], dt)
        ghks_results = sgp_smoother(m_and_cov, sgps, ghkf_results[0], ghkf_results[1], dt)
        cd_ghks_results = cd_sgp_smoother(drift, B, sgps, cd_ghkf_results[0], cd_ghkf_results[1], dt)

        for i in range(2):
            npt.assert_allclose(rts_results[i], eks_results[i])
            npt.assert_allclose(rts_results[i], ghks_results[i])
            npt.assert_allclose(rts_results[i], cd_eks_results[i], atol=1e-1)
            npt.assert_allclose(cd_eks_results[i], cd_ghks_results[i])


class TestBruteForce:

    @staticmethod
    def setting():
        # Times
        dt = 1e-2
        T = 100
        ts = jnp.linspace(dt, dt * T, T)
        measurement_noise_var = 0.1

        # Spatial grids
        xs = jnp.linspace(-5, 5, 1000)

        # SDE model
        ell, sigma = 1., 0.5
        mean0, var0 = 0., sigma ** 2
        init_ps = jax.scipy.stats.norm.pdf(xs, mean0, math.sqrt(var0))

        def drift(x):
            return -1 / ell * x

        def dispersion(_):
            return math.sqrt(2) * sigma / math.sqrt(ell)

        # Explicit discretisation of the SDE
        F, Sigma = math.exp(-dt / ell), sigma ** 2 * (1 - math.exp(-2 * dt / ell))

        def matern12(t1, t2):
            return jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell) * sigma ** 2

        def measurement_cond_pdf(y, x):
            """p(y | x)
            """
            return jax.scipy.stats.norm.pdf(y, x, math.sqrt(measurement_noise_var))

        def kf(_ys):
            def scan_body(carry, elem):
                mf, vf, nell = carry
                y = elem

                mp = F * mf
                vp = F * vf * F + Sigma

                s = vp + measurement_noise_var
                k = vp / s
                mf = mp + k * (y - mp)
                vf = vp - vp * k
                nell -= jax.scipy.stats.norm.logpdf(y, mp, jnp.sqrt(s))
                return (mf, vf, nell), (mf, vf)

            (*_, nell_ys), (mfs, vfs) = jax.lax.scan(scan_body, (mean0, var0, 0.), _ys)
            return mfs, vfs, nell_ys

        key = jax.random.PRNGKey(666)

        key, _ = jax.random.split(key)
        traj = jnp.linalg.cholesky(matern12(ts, ts)) @ jax.random.normal(key, (T,))

        key, _ = jax.random.split(key)
        ys = traj + math.sqrt(measurement_noise_var) * jax.random.normal(key, (T,))

        true_mfs, true_vfs, _ = kf(ys)

        return drift, dispersion, measurement_cond_pdf, init_ps, xs, ys, dt, true_mfs, true_vfs

    def test_kolmogorov(self):
        drift, dispersion, measurement_cond_pdf, init_ps, xs, ys, dt, true_mfs, true_vfs = self.setting()
        pss = brute_force_filter(drift, dispersion, measurement_cond_pdf, init_ps, xs, ys, dt, integration_steps=20,
                                 pred_method='kolmogorov')

        estimated_means = jnp.trapz(pss * xs[None, :], xs, axis=1)
        estimated_2nd_ms = jnp.trapz(pss * xs[None, :] ** 2, xs, axis=1)

        npt.assert_allclose(estimated_means, true_mfs, atol=1e-4, rtol=1e-2)
        npt.assert_allclose(estimated_2nd_ms, true_vfs + true_mfs ** 2, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize('method', ['chapman-euler',
                                        'chapman-tme-2',
                                        'chapman-tme-3'])
    def test_chapman(self, method):
        drift, dispersion, measurement_cond_pdf, init_ps, xs, ys, dt, true_mfs, true_vfs = self.setting()
        pss = brute_force_filter(drift, dispersion, measurement_cond_pdf, init_ps, xs, ys, dt, integration_steps=20,
                                 pred_method=method)

        estimated_means = jnp.trapz(pss * xs[None, :], xs, axis=1)
        estimated_2nd_ms = jnp.trapz(pss * xs[None, :] ** 2, xs, axis=1)

        if method == 'chapman-euler':
            atol, rtol = 1e-4, 1e-2
        elif method == 'chapman-tme-2':
            atol, rtol = 1e-7, 1e-6
        elif method == 'chapman-tme-3':
            atol, rtol = 1e-11, 1e-10
        else:
            raise ValueError(f'What is {method}?')

        npt.assert_allclose(estimated_means, true_mfs, atol=atol, rtol=rtol)
        npt.assert_allclose(estimated_2nd_ms, true_vfs + true_mfs ** 2, atol=atol, rtol=rtol)

        zs = jnp.linspace(-100, 100, 100)
        true_cfs = jnp.exp(1.j * zs[None, :] * true_mfs[:, None] - 0.5 * true_vfs[:, None] * zs[None, :] ** 2)
        estimated_cfs = jax.vmap(jax.vmap(characteristic_from_pdf,
                                          in_axes=[0, None, None]),
                                 in_axes=[None, 0, None])(zs, pss, xs)

        npt.assert_allclose(true_cfs, estimated_cfs, atol=atol, rtol=rtol)
