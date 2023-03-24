import pytest
import sympy
import math
import jax
import jax.scipy
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from mfs.utils import GaussianSum1D, vmap_list_of_funcs, partial_bell, complete_bell, hermite_probabilist, lanczos, \
    lanczos_ritz, GaussianSumND, ldl, ldl_chol
from mfs.utils import discretise_lti_sde, posterior_cramer_rao
from mfs.one_dim.moments import raw_to_central, raw_to_scaled
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, gram_and_hankel_indices_graded_lexico
from mfs.multi_dims.quadratures import moment_quadrature_nd
from mfs.classical_filters_smoothers import kf
from jax.config import config

config.update("jax_enable_x64", True)
np.random.seed(666)


class TestUtils:

    @pytest.mark.parametrize('N', [4, 5])
    def test_gaussian_sum_1d(self, N):
        means = jnp.array([-1.1, 1.2])
        variances = jnp.array([0.1, 0.1])
        weights = jnp.array([0.4, 0.6])

        gs = GaussianSum1D.new(means, variances, weights, N)

        # Test moments
        npt.assert_allclose(gs.cms, raw_to_central(gs.rms), atol=1e-12, rtol=1e-15)
        npt.assert_allclose(gs.scms, raw_to_scaled(gs.rms), atol=1e-12, rtol=1e-15)

        # Test pdf
        xs = jnp.linspace(-5, 5, 1000)
        npt.assert_allclose(jnp.trapz(gs.pdf(xs) * xs, xs), gs.mean)

        # Test sampling
        key = jax.random.PRNGKey(666)
        samples = gs.sampler(key, 100000)
        npt.assert_allclose(jnp.mean(samples), gs.mean, atol=1e-2, rtol=1e-2)

    def test_gaussian_sum_nd(self):
        d = 2
        N = 10

        multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        means = jnp.array([[2., 2.],
                           [-1., -1.],
                           [-3, 3]])
        covs = jnp.array([[[0.2, 0.1],
                           [0.1, 1.]],
                          [[2., 0.2],
                           [0.2, 0.3]],
                          [[0.5, 0.],
                           [0., 1.]]])
        weights = jnp.array([0.4, 0.4, 0.2])

        gs = GaussianSumND.new(means, covs, weights, multi_indices)

        key = jax.random.PRNGKey(666)

        samples = gs.sampler(key, 100000)

        # Test mean and cov
        npt.assert_allclose(np.mean(samples, axis=0), gs.mean, atol=1e-2, rtol=1e-1)
        npt.assert_allclose(np.cov(samples, rowvar=False), gs.cov, atol=1e-1, rtol=1e-1)

        # Test cf
        ws, nodes = moment_quadrature_nd(gs.cms, inds, gs.mean)

        z = np.random.randn(d)

        cf_mc = np.mean(np.exp(1.j * np.dot(samples, z)))
        cf_quad = np.dot(np.exp(1.j * np.dot(nodes, z)), ws)

        npt.assert_allclose(cf_mc, cf_quad, atol=1e-3, rtol=1e-2)

        # Test logpdf
        x = jnp.asarray(np.random.randn(d))
        npt.assert_allclose(gs.logpdf(x), np.log(gs.pdf(x)), atol=1e-12, rtol=1e-12)

    def test_vmap_funcs(self):
        funcs = [lambda z: jnp.sin(z),
                 lambda z: jnp.exp(z),
                 lambda z: z ** 2]
        vmap_func = vmap_list_of_funcs(funcs)

        x = jnp.array([1.1, 2.2, 3.3])
        results = vmap_func(x)
        for i in range(len(funcs)):
            npt.assert_array_equal(results[i], funcs[i](x))

    def test_lti_disc(self):
        lam, f, dt = 1.1, 2.2, 0.1
        A = jnp.array([[-lam, -2 * math.pi * f],
                       [2 * math.pi * f, -lam]])

        B = jnp.eye(2)
        F, Q = discretise_lti_sde(A, B, dt)

        z = 2 * math.pi * dt * f
        expected_F = jnp.array([[jnp.cos(z), -jnp.sin(z)],
                                [jnp.sin(z), jnp.cos(z)]]) * jnp.exp(-dt * lam)
        if lam == 0:
            expected_Q = jnp.eye(2) * dt
        else:
            expected_Q = jnp.eye(2) * (1 - jnp.exp(-2 * dt * lam)) / (2 * lam)

        npt.assert_allclose(F, expected_F)
        npt.assert_allclose(Q, expected_Q, atol=1e-12)

    @pytest.mark.parametrize('n', [5, 9])
    @pytest.mark.parametrize('k', [2, 4])
    def test_partial_bell(self, n, k):

        key = jax.random.PRNGKey(666)
        xs = jax.random.normal(key, (n - k + 1,))

        computed_result = jax.jit(partial_bell, static_argnums=(0, 1))(n, k, xs)
        theoretical_result = sympy.bell(n, k, np.array(xs)).evalf()
        npt.assert_almost_equal(computed_result, theoretical_result)

    def test_complete_bell(self):
        n = 4

        key = jax.random.PRNGKey(666)
        xs = jax.random.normal(key, (n,))

        computed_result = jax.jit(complete_bell, static_argnums=(0,))(n, xs)
        theoretical_result = jnp.array(xs[0] ** 4 + 6 * xs[0] ** 2 * xs[1] + 4 * xs[0] * xs[2] + 3 * xs[1] ** 2 + xs[3])
        npt.assert_almost_equal(computed_result, theoretical_result)

    def test_hermite(self):
        x = 2.2387

        for order in range(6):
            _c = [0.] * (order + 1)
            _c[-1] = 1.
            true_hermite_eval = np.polynomial.hermite_e.hermeval(x, _c)
            computed_hermite_eval = hermite_probabilist(order, x)
            npt.assert_almost_equal(computed_hermite_eval, true_hermite_eval)

    @pytest.mark.parametrize('dim', [5, 10])
    def test_lanczos(self, dim):
        num_iteration = 4

        key = jax.random.PRNGKey(666)
        A = jax.random.normal(key, (dim, dim))
        A = A @ A.T + 0.1 * jnp.eye(dim)
        v0 = np.zeros((dim,))
        v0[0] = 1.
        v0 = jnp.asarray(v0)

        vs, alphas, betas = lanczos(A, v0, num_iteration)
        T = jnp.diag(alphas) + jnp.diag(betas, k=-1) + jnp.diag(betas, k=1)
        npt.assert_allclose(T, vs.T @ A @ vs, atol=1e-10)

        # Test orthonormality of vs
        cov = np.zeros((num_iteration, num_iteration))
        for i in range(num_iteration):
            for j in range(num_iteration):
                cov[i, j] = jnp.dot(vs[:, i], vs[:, j]) / jnp.linalg.norm(vs[:, i], ord=2)
        npt.assert_allclose(cov, jnp.eye(num_iteration), atol=1e-9)

        # Should give exact eigenvals and eigenvecs when num_iteration=dim
        num_iteration = dim
        vs, alphas, betas = lanczos(A, v0, num_iteration)
        T = jnp.diag(alphas) + jnp.diag(betas, k=-1) + jnp.diag(betas, k=1)
        npt.assert_allclose(vs @ vs.T, vs.T @ vs, atol=1e-9)
        npt.assert_allclose(vs @ vs.T, jnp.eye(num_iteration), atol=1e-9)
        npt.assert_allclose(A, vs @ T @ vs.T)

        _, desired_eigenvals = jax.lax.linalg.eigh(A, sort_eigenvalues=True)
        _, lanczos_eigenvals = jax.lax.linalg.eigh(T, sort_eigenvalues=True)
        npt.assert_allclose(lanczos_eigenvals, desired_eigenvals)

    @pytest.mark.parametrize('dim', [5, 200])
    def test_lanczos_ritz(self, dim):
        key = jax.random.PRNGKey(666)
        A = jax.random.uniform(key, (dim, dim)) / dim
        A = A @ A.T + 0.1 * jnp.eye(dim)

        key, _ = jax.random.split(key)
        v0 = jax.random.normal(key, (dim,))

        ritz_vectors, ritz_values = lanczos_ritz(A, v0, dim)

        for n in range(dim - 1):
            theoretical_result = jnp.linalg.matrix_power(A, n) @ v0
            computed_result = jnp.einsum('i,...i->...', ritz_values ** n, ritz_vectors)
            npt.assert_allclose(computed_result, theoretical_result, rtol=1e-12, atol=1e-12)

    def test_ldl(self):
        a = np.random.randn(5, 5)
        a = a @ a.T

        l, d = ldl(a)
        npt.assert_allclose(jnp.linalg.cholesky(a), l @ jnp.diag(jnp.sqrt(d)))
        npt.assert_allclose(jnp.linalg.cholesky(a), ldl_chol(a))

        a = np.random.randn(5)
        a = np.outer(a, a) - 1e-8 * jnp.eye(5)
        c = ldl_chol(a)
        npt.assert_allclose(a, c @ c.T, rtol=1e-3)


class TestCramerRao:

    def test_crlb_lgssm(self):
        ell, sigma = 1., 1.
        dt = 0.1
        T = 10

        A = jnp.array([[0., 1.],
                       [-3 / ell ** 2, -2 * math.sqrt(3) / ell]])
        B = jnp.array([[0., 0.],
                       [0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5]])

        F, Sigma = discretise_lti_sde(A, B, dt)
        chol_sigma = jnp.linalg.cholesky(Sigma)

        Xi = 1.
        H = jnp.array([[1., 0.]])

        m0 = jnp.zeros((2,))
        P0 = jnp.asarray(np.diag([sigma ** 2, 3 / ell ** 2 * sigma ** 2]))
        chol_P0 = jnp.sqrt(P0)

        num_mcs = 1000000

        def simulate(key):
            def scan_body(carry, elem):
                x = carry
                rnd_x, rnd_y = elem
                x = F @ x + chol_sigma @ rnd_x
                y = H @ x + math.sqrt(Xi) * rnd_y
                return x, (x, y)

            rnds_x = jax.random.normal(key, shape=(T, 2))
            key, _ = jax.random.split(key)
            rnds_y = jax.random.normal(key, shape=(T,))
            key, _ = jax.random.split(key)
            x0 = m0 + chol_P0 @ jax.random.normal(key, shape=(2,))
            _, (xs, ys) = jax.lax.scan(scan_body, x0, (rnds_x, rnds_y))
            return jnp.concatenate([x0[None, :], xs], axis=0), ys

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num_mcs)
        xss, yss = jax.vmap(simulate, in_axes=[0])(keys)

        kf_results = jax.vmap(kf, in_axes=[None, None, None, None, None, None, 0])(F, Sigma, H, jnp.atleast_2d(Xi),
                                                                                   m0, P0, yss)
        mfs, Pfs, _ = kf_results

        # Test if Pfs are the same for all MC runs
        i, k = np.random.randint(0, T - 1, 2)
        npt.assert_array_equal(Pfs[i], Pfs[k])

        Pfs = Pfs[0]

        # Test if E[(mfs - x)(mfs - x)^T] is all close to Pfs
        res = (mfs - xss[:, 1:, :])
        E = jnp.mean(jnp.einsum('...i,...j->...ij', res, res), axis=0)
        npt.assert_allclose(E, Pfs, atol=1e-1)

        # Test if CRLB is all close to Pfs
        def logpdf_transition(xt, xs):
            return jax.scipy.stats.multivariate_normal.logpdf(xt, F @ xs, Sigma)

        def logpdf_likelihood(yt, xt):
            return jnp.squeeze(jax.scipy.stats.norm.logpdf(yt, H @ xt, math.sqrt(Xi)))

        xss = jnp.transpose(xss, [1, 0, 2])
        yss = jnp.transpose(yss, [1, 0, 2])

        j0 = jnp.linalg.inv(P0)
        js = posterior_cramer_rao(xss, yss, j0, logpdf_transition, logpdf_likelihood)
        npt.assert_allclose(jnp.linalg.inv(js), Pfs, atol=1e-12)
