"""
Test the filtering routines.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
import numpy.testing as npt
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, gram_and_hankel_indices_graded_lexico
from mfs.one_dim.filtering import moment_filter_rms, moment_filter_cms, moment_filter_scms
from mfs.one_dim.moments import raw_to_scaled, raw_moment_of_normal, raw_to_central
from mfs.multi_dims.moments import central_moments_mvn_kan, raw_moments_mvn_kan, marginalise_moments
from mfs.multi_dims.filtering import moment_filter_nd_rms, moment_filter_nd_cms, moment_filter_nd_scms
from jax.config import config
from functools import partial

np.random.seed(666)
config.update("jax_enable_x64", True)

# Times
dt = 1e-2
T = 100
ts = jnp.linspace(dt, dt * T, T)

# Model parameters
ell, sigma = 1., 0.5


def matern12(t1, t2):
    return jnp.exp(-jnp.abs(t1[None, :] - t2[:, None]) / ell) * sigma ** 2


# Generate measurements
measurement_noise_var = 1.
ys = jnp.asarray(jnp.linalg.cholesky(matern12(ts, ts)) @ np.random.randn(T)
                 + math.sqrt(measurement_noise_var) * np.random.randn(T))
ys_2d = jnp.concatenate([ys[:, None], ys[:, None]], axis=-1)


def measurement_cond_pdf(y, x):
    return jnp.squeeze(jax.scipy.stats.norm.pdf(y, x, math.sqrt(measurement_noise_var)))


def measurement_cond_pdf_2d(y, x):
    return math.prod(jax.scipy.stats.norm.pdf(y, x, math.sqrt(measurement_noise_var)))


def drift(x):
    return -x / ell


def dispersion(_):
    return math.sqrt(2) * sigma / math.sqrt(ell)


def dispersion_2d(_):
    return math.sqrt(2) * sigma / math.sqrt(ell) * jnp.eye(2)


def kf(F, Sigma, mean0, var0):
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

    (*_, nell_ys), (mfs, vfs) = jax.lax.scan(scan_body, (mean0, var0, 0.), ys)
    return mfs, vfs, nell_ys


class Test1DFiltering:

    def test_1d_convergence(self):
        """The moment filter is asymptotically exact. This tests if the solution converges to that of the Kalman filter
        for the linear model.
        """
        tme_order = 3

        F, Sigma = math.exp(-1 * dt / ell), sigma ** 2 * (1 - math.exp(-2 / ell * dt))

        @partial(jax.vmap, in_axes=[0, None])
        @partial(jax.vmap, in_axes=[None, 0])
        def state_cond_raw_moments(x, n):
            def phi(u):
                return u ** n

            return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

        # Sometimes a non-zero mean can initialise a numerical bug due to TME,
        # see https://github.com/zgbkdlm/tme/issues/9
        mean0 = 0.1
        var0 = 0.1

        N = 10
        rms0 = jnp.array([raw_moment_of_normal(mean0, var0, p) for p in range(2 * N)])

        rmss, nell_ys = moment_filter_rms(state_cond_raw_moments, measurement_cond_pdf, rms0, ys)
        true_mfs, true_vfs, true_nell_ys = kf(F, Sigma, mean0, var0)

        npt.assert_allclose(rmss[:, 1], true_mfs, rtol=1e-2)
        npt.assert_allclose(rmss[:, 2] - rmss[:, 1] ** 2, true_vfs, rtol=1e-3)
        npt.assert_allclose(nell_ys, true_nell_ys, rtol=1e-5)

    def test_routines_equivalence(self):
        """Test if the moment filtering using rms, cms, and scms are the same. Technically, they should not be the same
        due to integration and rounding errors, but they should be super close.
        """
        tme_order = 2
        N = 4

        mean0 = 0.
        var0 = 0.5
        scale0 = math.sqrt(var0)

        rms0 = jnp.array([raw_moment_of_normal(mean0, var0, p) for p in range(2 * N)])
        cms0 = raw_to_central(rms0)
        scms0 = raw_to_scaled(rms0)

        @partial(jax.vmap, in_axes=[0, None, None, None])
        @partial(jax.vmap, in_axes=[None, 0, None, None])
        def state_cond_scaled_central_moments(x, n, mean, scale):
            def phi(u):
                return ((u - mean) / scale) ** n

            return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

        def state_cond_raw_moments(x, n):
            return state_cond_scaled_central_moments(x, n, 0., 1.)

        def state_cond_central_moments(x, n, mean):
            return state_cond_scaled_central_moments(x, n, mean, 1.)

        @partial(jax.vmap, in_axes=[0])
        def state_cond_mean(x):
            return jnp.squeeze(tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)[0])

        @partial(jax.vmap, in_axes=[0])
        def state_cond_mean_var(x):
            cond_m, cond_var = tme.mean_and_cov(jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order)
            return jnp.squeeze(cond_m), jnp.squeeze(cond_var)

        rmss, nell_r = moment_filter_rms(state_cond_raw_moments, measurement_cond_pdf, rms0, ys)
        cmss, means_c, nell_ys_c = moment_filter_cms(state_cond_central_moments, state_cond_mean,
                                                     measurement_cond_pdf, cms0, mean0, ys)
        scmss, means, scales, nell_ys_s = moment_filter_scms(state_cond_scaled_central_moments,
                                                             state_cond_mean_var,
                                                             measurement_cond_pdf, scms0, mean0, scale0, ys)

        npt.assert_array_almost_equal(cmss, jax.vmap(raw_to_central, in_axes=[0])(rmss), decimal=11)
        npt.assert_array_almost_equal(scmss, jax.vmap(raw_to_scaled, in_axes=[0])(rmss), decimal=10)
        npt.assert_array_almost_equal(means_c, means, decimal=15)
        npt.assert_array_almost_equal(rmss[:, 2] - rmss[:, 1] ** 2, scales ** 2, decimal=12)

        for nell in [nell_ys_s, nell_ys_c]:
            npt.assert_array_almost_equal(nell_r, nell, decimal=11)


class TestNDFiltering:

    def test_routines_equivalence(self):
        """ND filters should be numerically the same with different moment modes.
        """
        d = 2
        tme_order = 2
        N = 3

        multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
        multi_indices_jax = jnp.asarray(multi_indices)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        @partial(jax.vmap, in_axes=[0, None, None, None])
        @partial(jax.vmap, in_axes=[None, 0, None, None])
        def state_cond_scaled_central_moments(x, index, mean, scale):
            cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion_2d, order=tme_order)
            list_raw_moments = jnp.array(
                [raw_moments_mvn_kan(cond_mean - mean, cond_cov, multi_index) for multi_index in multi_indices])
            s = jnp.prod(scale ** multi_indices_jax[index])
            return list_raw_moments[index] / s

        def state_cond_raw_moments(x, index):
            return state_cond_scaled_central_moments(x, index, 0., 1.)

        def state_cond_central_moments(x, index, mean):
            return state_cond_scaled_central_moments(x, index, mean, 1.)

        @partial(jax.vmap, in_axes=[0])
        def state_cond_mean(x):
            return tme.mean_and_cov(x, dt, drift, dispersion_2d, order=tme_order)[0]

        @partial(jax.vmap, in_axes=[0])
        def state_cond_mean_var(x):
            cond_mean, cond_cov = tme.mean_and_cov(x, dt, drift, dispersion_2d, order=tme_order)
            return cond_mean, jnp.diag(cond_cov)

        mean0 = jnp.array([1., 1.])
        cov0 = jnp.eye(d)
        scale0 = jnp.sqrt(jnp.diag(cov0))
        rms0 = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(np.asarray(mean0), np.asarray(cov0),
                                                                                multi_indices)
        cms0 = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(np.asarray(cov0), multi_indices)
        scms0 = cms0 / jnp.array([math.prod(scale0 ** multi_index) for multi_index in multi_indices])

        rmss, nell_r = moment_filter_nd_rms((state_cond_raw_moments, 'index'),
                                            measurement_cond_pdf_2d,
                                            ys_2d,
                                            (multi_indices, inds),
                                            rms0)
        cmss, means_c, nell_c = moment_filter_nd_cms((state_cond_central_moments, 'index'),
                                                     state_cond_mean,
                                                     measurement_cond_pdf_2d, ys_2d,
                                                     (multi_indices, inds),
                                                     cms0, mean0)
        scmss, means_s, scales_s, nell_s = moment_filter_nd_scms((state_cond_scaled_central_moments, 'index'),
                                                                 state_cond_mean_var,
                                                                 measurement_cond_pdf_2d,
                                                                 ys_2d,
                                                                 (multi_indices, inds),
                                                                 scms0, mean0, scale0)

        # Test equivalences
        npt.assert_allclose(means_s, means_c, atol=1e-12, rtol=1e-12)

        npt.assert_allclose(rmss[:, 1], means_c[:, 1], atol=1e-10, rtol=1e-8)
        npt.assert_allclose(rmss[:, 2], means_c[:, 0], atol=1e-10, rtol=1e-8)

        npt.assert_allclose(rmss[:, 3] - rmss[:, 1] ** 2, scales_s[:, 1] ** 2, atol=1e-11, rtol=1e-10)
        npt.assert_allclose(rmss[:, 5] - rmss[:, 2] ** 2, scales_s[:, 0] ** 2, atol=1e-11, rtol=1e-10)

        for n, multi_index in zip(range(multi_indices.shape[0]), multi_indices):
            npt.assert_allclose(cmss[:, n], scmss[:, n] * np.prod(scales_s ** multi_index, axis=1), atol=1e-14)

        for nell in [nell_c, nell_s]:
            npt.assert_array_almost_equal(nell_r, nell, decimal=11)

    def test_reduce_to_1d(self):
        """The ND filtering with independent states and measurements should boil down to copies of 1D filtering.
        """
        tme_order = 3
        N = 3

        m0 = 0.1
        var0 = 0.2

        # 1D filtering
        @partial(jax.vmap, in_axes=[0, None])
        @partial(jax.vmap, in_axes=[None, 0])
        def state_cond_raw_moments_1d(x, n):
            def phi(u):
                return u ** n

            return jnp.squeeze(tme.expectation(phi, jnp.atleast_1d(x), dt, drift, dispersion, order=tme_order))

        rms0_1d = jnp.array([raw_moment_of_normal(m0, var0, p) for p in range(2 * N)])
        rmss_1d, nell_1d = moment_filter_rms(state_cond_raw_moments_1d, measurement_cond_pdf, rms0_1d, ys)

        # 2D filtering
        d = 2

        multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        @partial(jax.vmap, in_axes=[0, None])
        @partial(jax.vmap, in_axes=[None, 0])
        def state_cond_raw_moments(x, multi_index):
            def phi(u):
                return jnp.prod(u ** multi_index)

            return tme.expectation(phi, x, dt, drift, dispersion_2d, order=tme_order)

        mean0 = m0 * jnp.ones((d,))
        cov0 = var0 * jnp.eye(d)
        rms0 = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(np.asarray(mean0), np.asarray(cov0),
                                                                                multi_indices)

        rmss_2d, nell_2d = moment_filter_nd_rms((state_cond_raw_moments, 'multi-index'),
                                                measurement_cond_pdf_2d,
                                                ys_2d,
                                                (multi_indices, inds),
                                                rms0)

        # Initial
        npt.assert_allclose(rms0_1d, marginalise_moments(rms0, d, N, 0))

        # All dims should be the same
        npt.assert_allclose(marginalise_moments(rmss_2d, d, N, 0), marginalise_moments(rmss_2d, d, N, 1))

        # Should be the same as the 1D filter.
        # rtol=1e-3 show the numerical errors due to e.g, Cholesky, eigh, and linear_solve
        npt.assert_allclose(rmss_1d, marginalise_moments(rmss_2d, d, N, 0), rtol=1e-3)

        # Likelihood should be the same
        # TODO: Why do the numerical errors affect the moment estimation above but not the likelihood?
        npt.assert_allclose(nell_1d * 2, nell_2d)

        # Running the Nd filtering with d = 1 should have exactly the same implementation as the 1D filtering
        d = 1

        @partial(jax.vmap, in_axes=[0, None])
        @partial(jax.vmap, in_axes=[None, 0])
        def state_cond_raw_moments_ext(x, multi_index):
            def phi(u):
                return jnp.prod(u ** multi_index)

            return tme.expectation(phi, x, dt, drift, dispersion, order=tme_order)

        multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        mean0 = m0 * jnp.ones((d,))
        cov0 = var0 * jnp.eye(d)
        rms0 = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(np.asarray(mean0), np.asarray(cov0),
                                                                                multi_indices)

        rmss_ext, nell_ext = moment_filter_nd_rms((state_cond_raw_moments_ext, 'multi-index'),
                                                  measurement_cond_pdf,
                                                  ys,
                                                  (multi_indices, inds),
                                                  rms0)
        npt.assert_allclose(rmss_ext, rmss_1d)
        npt.assert_allclose(nell_ext, nell_1d)
