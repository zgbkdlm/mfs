import numpy as np
import jax.numpy as jnp
import math
import pytest
import numpy.testing as npt
from mfs.one_dim.moments import raw_moment_of_normal
from mfs.multi_dims.moments import central_moments_mvn_kan, raw_moments_mvn_mgf, raw_moments_mvn_kan, extract_mean, \
    extract_cov, sde_cond_moments_euler_maruyama, sde_cond_moments_tme_normal, sde_cond_moments_tme, marginalise_moments
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, gram_and_hankel_indices_graded_lexico
from mfs.utils import discretise_lti_sde
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)


class TestMultiDimMoments:

    def test_nd_vs_independent_1ds(self):
        d = 3

        mean = np.random.randn(d)
        cov = np.diag(np.random.randn(d) ** 2)

        for i in range(d):
            multi_index = [0] * d
            multi_index[i] = 2

            rm = raw_moments_mvn_kan(mean, cov, multi_index)
            rm2 = raw_moment_of_normal(mean[i], cov[i, i], 2)
            npt.assert_array_equal(rm, rm2)

    @pytest.mark.parametrize('d, multi_index', [(1, [2]),
                                                (3, [0, 0, 0]),
                                                (3, [1, 1, 1]),
                                                (3, [1, 2, 1]),
                                                (4, [2, 1, 1, 3])])
    def test_mvn_central_moments(self, d, multi_index):
        _c = np.random.randn(d, d)
        cov = _c @ _c.T + np.eye(d)

        computed_cm_kan = central_moments_mvn_kan(cov, multi_index)
        computed_cm_mgf = raw_moments_mvn_mgf(np.zeros((d,)), cov, multi_index)
        npt.assert_almost_equal(computed_cm_kan, computed_cm_mgf)

        mean = np.random.randn(d)
        computed_cm_kan = raw_moments_mvn_kan(mean, cov, multi_index)
        computed_cm_mgf = raw_moments_mvn_mgf(mean, cov, multi_index)
        npt.assert_almost_equal(computed_cm_kan, computed_cm_mgf)

    @pytest.mark.parametrize('d, multi_index', [(1, [2]),
                                                (3, [0, 0, 0]),
                                                (3, [1, 2, 1]),
                                                (4, [5, 2, 1, 0]),
                                                (4, [3, 2, 1, 1])])
    def test_kan_moments(self, d, multi_index):
        mean = np.zeros((d,))
        _c = np.random.randn(d, d)
        cov = _c @ _c.T + np.eye(d)

        rms = raw_moments_mvn_kan(mean, cov, multi_index)
        cms = central_moments_mvn_kan(cov, multi_index)
        npt.assert_array_almost_equal(rms, cms)

        mean = np.random.randn(d)
        for i in range(d):
            for j in range(d):
                multi_index = [0] * d
                multi_index[i] = 1
                multi_index[j] += 1
                ijth_second_moment = raw_moments_mvn_kan(mean, cov, multi_index)
                npt.assert_array_almost_equal(ijth_second_moment - mean[i] * mean[j], cov[i, j])

    @pytest.mark.parametrize('d, multi_index', [(1, [2]),
                                                (2, [1, 3]),
                                                (3, [3, 1, 2])])
    def test_monte_carlo(self, d, multi_index):
        mean = np.random.randn(d)
        a = np.random.randn(d)
        cov = np.outer(a, a) + np.eye(d)

        def f(x):
            return np.prod([elem[None, :] ** power for elem, power in zip(x, multi_index)], axis=0)

        num_mcs = 100000
        xs = mean[:, None] + np.linalg.cholesky(cov) @ np.random.randn(d, num_mcs)
        rms_mc = np.mean(f(xs))
        rms = raw_moments_mvn_kan(mean, cov, multi_index)
        npt.assert_allclose(rms, rms_mc, rtol=3.1e-2)

    @pytest.mark.parametrize('N', [3, 5])
    @pytest.mark.parametrize('d', [1, 3, 4])
    def test_gram_matrices_generation(self, N, d):
        multi_indices_of_cms = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
        cms = np.zeros((multi_indices_of_cms.shape[0],))

        inds = gram_and_hankel_indices_graded_lexico(N, d)

        _c = np.random.randn(d, d)
        cov = _c @ _c.T + np.eye(d)
        for i, multi_index in enumerate(multi_indices_of_cms):
            cms[i] = central_moments_mvn_kan(cov, multi_index)

        # Test if indexable and symmetric
        for ind in inds:
            mat = cms[ind]
            npt.assert_array_equal(mat, mat.T)

        # Test p.d.
        gram = cms[inds[0]]
        eigenvals = np.linalg.eigvalsh(gram)
        assert np.min(eigenvals) > 0.

    def test_extraction(self):
        d = 2
        N = 5

        mean = np.random.randn(d)
        _c = np.random.randn(d, d)
        cov = _c @ _c.T + np.eye(d)

        multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
        rms = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean, cov, multi_indices)
        cms = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(cov, multi_indices)

        npt.assert_array_almost_equal(extract_mean(rms, d), mean, decimal=20)
        npt.assert_array_almost_equal(extract_cov(cms, d), cov, decimal=15)

        rms_ext = np.tile(rms, (2, 3, 1))
        cms_ext = np.tile(cms, (2, 3, 1))

        npt.assert_array_almost_equal(extract_mean(rms_ext, d), np.tile(mean, (2, 3, 1)), decimal=20)
        npt.assert_array_almost_equal(extract_cov(cms_ext, d), np.tile(cov, (2, 3, 1, 1)), decimal=15)

    @pytest.mark.parametrize('N', [3, 4])
    @pytest.mark.parametrize('d', [2, 3])
    def test_marginalisation(self, d, N):
        mean = np.random.randn(d)
        cov = np.diag(np.random.rand(d) ** 2)

        multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
        joint_rms = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean, cov, multi_indices)

        for _d in range(d):
            marginal_rms = jnp.array([raw_moment_of_normal(mean[_d], cov[_d, _d], p) for p in range(2 * N)])
            npt.assert_allclose(marginalise_moments(joint_rms, d, N, _d), marginal_rms)

    def test_transition_cond(self):
        """Test the condition conditional moments.
        """
        d = 2
        N = 4
        tme_order = 3

        multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
        inds = gram_and_hankel_indices_graded_lexico(N, d)
        length = multi_indices.shape[0]

        dt = 0.01

        # Model
        ell, sigma = 1., 1.

        A = jnp.array([[0., 1.],
                       [-3 / ell ** 2, -2 * math.sqrt(3) / ell]])
        B = jnp.array([[1., 0.],
                       [0., 2 * (math.sqrt(3) / ell) ** 1.5 * sigma]])

        def drift(x):
            return A @ x

        def dispersion(_):
            return B

        F, Sigma = discretise_lti_sde(A, B, dt)
        init_x = jnp.asarray(np.random.randn(d))

        true_mean = F @ init_x
        true_cov = Sigma
        true_scale = jnp.sqrt(jnp.diag(true_cov))
        true_cms = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(np.asarray(true_cov), multi_indices)
        true_rms = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(np.asarray(true_mean),
                                                                                    np.asarray(true_cov),
                                                                                    multi_indices)
        true_scms = true_cms / jnp.array([math.prod(true_scale ** multi_index) for multi_index in multi_indices])

        _, cond_cms_em, cond_scms_em, cond_mean_em, cond_mean_var_em = sde_cond_moments_euler_maruyama(drift,
                                                                                                       dispersion,
                                                                                                       dt,
                                                                                                       multi_indices)
        _, cond_cms_tme_normal, cond_scms_tme_normal, cond_mean_tme_normal, cond_mean_var_tme_normal = sde_cond_moments_tme_normal(
            drift,
            dispersion,
            dt,
            tme_order,
            multi_indices)
        _, cond_cms_tme, cond_scms_tme, cond_mean_tme, cond_mean_var_tme = sde_cond_moments_tme(drift, dispersion,
                                                                                                dt, tme_order)

        init_x = init_x.reshape(1, -1)

        # Test means
        mean_em = cond_mean_em(init_x)[0]
        mean_tme_normal = cond_mean_tme_normal(init_x)[0]
        mean_tme = cond_mean_tme(init_x)[0]

        npt.assert_allclose(mean_em, true_mean, rtol=1e-2)
        npt.assert_allclose(mean_tme_normal, true_mean, rtol=1e-4)
        npt.assert_array_equal(mean_tme_normal, mean_tme)

        mean_em2, var_em = cond_mean_var_em(init_x)
        mean_tme_normal2, var_tme_normal = cond_mean_var_tme_normal(init_x)
        mean_tme2, var_tme = cond_mean_var_tme(init_x)

        npt.assert_array_equal(mean_em2[0], mean_em)
        npt.assert_array_equal(mean_tme_normal2[0], mean_tme_normal)
        npt.assert_array_equal(mean_tme2[0], mean_tme)

        # Test scales
        npt.assert_allclose(var_em[0], true_scale ** 2, rtol=4e-2)
        npt.assert_allclose(var_tme_normal[0], true_scale ** 2, rtol=1e-4)
        npt.assert_allclose(var_tme[0], true_scale ** 2, rtol=1e-4)

        # Test central moments
        cms_em = cond_cms_em(init_x, jnp.arange(length), mean_em)[0]
        cms_tme_normal = cond_cms_tme_normal(init_x, jnp.arange(length), mean_tme_normal)[0]
        cms_tme = cond_cms_tme(init_x, jnp.asarray(multi_indices), mean_tme)[0]

        npt.assert_allclose(cms_em, true_cms, atol=1)
        npt.assert_allclose(cms_tme_normal, true_cms, rtol=1e-3)
        npt.assert_allclose(cms_tme, true_cms, atol=2.5)

        # Test scaled central moments
        scms_em = cond_scms_em(init_x, jnp.arange(length), mean_em, jnp.sqrt(var_em[0]))[0]
        scms_tme_normal = cond_scms_tme_normal(init_x, jnp.arange(length),
                                               mean_tme_normal, jnp.sqrt(var_tme_normal[0]))[0]
        # scms_tme = cond_scms_tme(init_x, jnp.arange(length), mean_tme, scale_tme[0])[0]

        npt.assert_allclose(scms_em, true_scms, atol=1)
        npt.assert_allclose(scms_tme_normal, true_scms, rtol=1e-3)
        # npt.assert_allclose(scms_tme, true_scms)

        # Test positive definiteness
        for G in [cms_em[inds[0]], cms_tme_normal[inds[0]], scms_em[inds[0]], scms_tme_normal[inds[0]]]:
            assert ~np.any(np.isnan(np.linalg.cholesky(G)))
