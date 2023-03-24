import math

import numpy as np
import pytest
import jax.numpy as jnp
import numpy.testing as npt
from mfs.one_dim.moments import raw_to_central, central_to_raw, raw_moment_of_normal, sms_to_cumulants, raw_to_scaled, \
    sde_cond_moments_tme, sde_cond_moments_tme_normal, central_moment_of_normal
from mfs.multi_dims.moments import sde_cond_moments_tme as sde_cond_moments_tme_nd
from jax.config import config
from mfs.utils import discretise_lti_sde

np.random.seed(666)
config.update("jax_enable_x64", True)


class TestMoments:

    @pytest.mark.parametrize('mean', [0., 1.1, 2.2])
    @pytest.mark.parametrize('variance', [4.9, 5.12])
    def test_conversion_to_central_moments(self, mean: float, variance: float):
        rms = jnp.array([raw_moment_of_normal(mean, variance, p) for p in range(4 + 1)])
        cms = raw_to_central(rms)

        theoretical_cms = jnp.array([1.,
                                     0.,
                                     rms[2] - mean ** 2,
                                     rms[3] - 3 * mean * rms[2] + 2 * mean ** 3,
                                     rms[4] - 4 * mean * rms[3] + 6 * mean ** 2 * rms[2] - 3 * mean ** 4])
        npt.assert_array_almost_equal(cms, theoretical_cms)

        # Central to non-central
        npt.assert_array_almost_equal(central_to_raw(theoretical_cms, mean), rms)

    @pytest.mark.parametrize('max_order', [4, ])
    @pytest.mark.parametrize('lam', [0.11, 2.22])
    def test_cumulants_exponential(self, max_order, lam):
        mean, variance = 1 / lam, 1 / lam ** 2
        rms = jnp.array([math.factorial(n) / lam ** n for n in range(max_order + 1)])
        cms = raw_to_central(rms)
        sms = raw_to_scaled(rms)

        computed_cumulants_1 = sms_to_cumulants(sms, mean, math.sqrt(variance))
        computed_cumulants_2 = sms_to_cumulants(cms, mean, 1.)
        computed_cumulants_3 = sms_to_cumulants(rms, 0., 1.)
        theoretical_cumulants = [rms[1],
                                 rms[2] - rms[1] ** 2,
                                 rms[3] - 3 * rms[2] * rms[1] + 2 * rms[1] ** 3,
                                 rms[4] - 4 * rms[3] * rms[1] - 3 * rms[2] ** 2 + 12 * rms[2] * rms[1] ** 2
                                 - 6 * rms[1] ** 4]
        for computed_cumulants in [computed_cumulants_1, computed_cumulants_2, computed_cumulants_3]:
            npt.assert_allclose(computed_cumulants, theoretical_cumulants, atol=1e-10, rtol=1e-12)


class TestCondMomentsGeneration:

    @staticmethod
    def setting():
        def drift(x):
            return -1.1 * x

        def dispersion(_):
            return 1.

        dt = 0.01
        return dt, drift, dispersion

    def test_equivalence_to_nd(self):
        dt, drift, dispersion = self.setting()

        tme_order = 2
        N = 3

        cond_rms, cond_cms, cond_scms, cond_mean, cond_mean_var = sde_cond_moments_tme(drift, dispersion, dt, tme_order)
        cond_rms_nd, cond_cms_nd, cond_scms_nd, cond_mean_nd, cond_mean_var_nd = sde_cond_moments_tme_nd(drift,
                                                                                                         dispersion,
                                                                                                         dt, tme_order)

        x = jnp.array([[1.1],
                       [2.2]])
        ns = jnp.arange(2 * N)

        npt.assert_array_equal(cond_rms(x, ns), cond_rms_nd(x, ns))
        npt.assert_array_equal(cond_cms(x, ns, 0.1), cond_cms_nd(x, ns, 0.1))
        npt.assert_array_equal(cond_scms(x, ns, 0.1, 0.2), cond_scms_nd(x, ns, 0.1, 0.2))
        npt.assert_array_equal(cond_mean(x), jnp.squeeze(cond_mean_nd(x)))
        for i1, i2 in zip(cond_mean_var(x), cond_mean_var_nd(x)):
            npt.assert_array_equal(i1, jnp.squeeze(i2))

    def test_normal_approx(self):
        dt, drift, dispersion = self.setting()

        tme_order = 3
        N = 6

        cond_rms, cond_cms, cond_scms, cond_mean, cond_mean_var = sde_cond_moments_tme(drift, dispersion, dt, tme_order)
        cond_rms_norm, cond_cms_norm, cond_scms_norm, cond_mean_norm, cond_mean_var_norm = sde_cond_moments_tme_normal(
            drift, dispersion, dt, tme_order, N)

        x = jnp.array([1.])
        ns = jnp.arange(2 * N)

        mean, variance = discretise_lti_sde(-1.1 * jnp.eye(1), jnp.eye(1), dt)
        mean, variance = mean[0, 0] * x[0], variance[0, 0] * x[0]
        rms = jnp.array([raw_moment_of_normal(mean, variance, p) for p in range(2 * N)])
        cms = jnp.array([central_moment_of_normal(variance, p) for p in range(2 * N)])

        npt.assert_allclose(mean, cond_mean(x)[0])
        npt.assert_allclose(variance, cond_mean_var(x)[1][0], atol=1e-8, rtol=1e-6)

        npt.assert_allclose(mean, cond_mean_norm(x)[0])
        npt.assert_allclose(variance, cond_mean_var_norm(x)[1][0], atol=1e-8, rtol=1e-6)

        npt.assert_allclose(rms, cond_rms(x, ns)[0], atol=1e-4, rtol=1e-4)
        npt.assert_allclose(cms, cond_cms(x, ns, cond_mean(x)[0])[0], atol=1e-5)

        npt.assert_allclose(rms, cond_rms_norm(x, ns)[0], atol=1e-6, rtol=1e-6)
        npt.assert_allclose(cms, cond_cms_norm(x, ns, cond_mean(x)[0])[0], atol=1e-8, rtol=1e-5)

        # Note that scms are not tested.
