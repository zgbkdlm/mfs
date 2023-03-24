import pytest
import jax
import jax.scipy.stats
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from sympy.core import symbols
from sympy.integrals.intpoly import Polygon, polytope_integrate
from mfs.multi_dims.moments import central_moments_mvn_kan, raw_moments_mvn_kan, moments_nd_uniform
from mfs.multi_dims.multi_indices import gram_and_hankel_indices_graded_lexico, \
    generate_graded_lexico_multi_indices
from mfs.multi_dims.quadratures import nd_cartesian_prod_indices, nd_cartesian_prod, moment_quadrature_nd
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from mfs.one_dim.quadtures import moment_quadrature as oned_gauss_quadrature
from jax.config import config
from functools import partial

config.update("jax_enable_x64", True)

np.random.seed(1)


class TestNDCartesianCombs:

    def test_nd_cartesian_inds(self):
        d, n = 3, 5
        inds = nd_cartesian_prod_indices(d, n)

        assert inds.shape == (n ** d, d)

        # Check if all the indices are unique
        assert np.unique(inds, axis=0).shape == inds.shape

    def test_nd_cartesian_eigvals_like(self):
        d, n = 3, 5
        x = np.random.randn(d, n)

        inds = nd_cartesian_prod_indices(d, n)
        carts = nd_cartesian_prod(jnp.array(x))

        for ind in inds:
            arr = np.array([x[_d, ind[_d]] for _d in range(d)])
            assert np.any(np.all((arr == carts), axis=1))

    def test_nd_cartesian_eigvectors_like(self):
        d, n = 3, 5
        x = np.random.randn(d, n, n)

        inds = nd_cartesian_prod_indices(d, n)
        carts = nd_cartesian_prod(jnp.array(x))

        for ind in inds:
            arr = np.vstack([x[_d, :, ind[_d]] for _d in range(d)]).T
            assert np.any(np.all((arr == carts), axis=(1, 2)))


def make_nd_quadrature_rules(d, N):
    key = jax.random.PRNGKey(666)
    _c = jax.random.normal(key, (d, d)) * 0.1
    cov = _c @ _c.T + jnp.eye(d)
    mean = jnp.zeros((d,))

    multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
    rms = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(cov, multi_indices)
    inds = gram_and_hankel_indices_graded_lexico(N, d)

    weights, nodes = moment_quadrature_nd(rms, inds)
    return mean, cov, multi_indices, rms, weights, nodes


@pytest.mark.parametrize('N', [4, 5])
def test_vs_1d_gauss_quadrature(N):
    """The nd quadrature must agree with the 1d quadrature when d = 1.
    """
    d = 1
    _, _, _, rms, weights, nodes = make_nd_quadrature_rules(d, N)
    oned_weights, oned_nodes = oned_gauss_quadrature(rms)
    npt.assert_allclose(weights, oned_weights)
    npt.assert_allclose(oned_nodes, nodes[:, 0])


def monomials(x, multi_index):
    return np.prod(np.asarray([x[:, idx] ** power for idx, power in enumerate(multi_index)]), axis=0)


@pytest.mark.parametrize('N', [4, 5])
@pytest.mark.parametrize('d', [2])
class TestMultiDimQuadratureNormal:

    def test_moments(self, d, N):
        """Test E[x^n].
        """
        _, _, multi_indices, rms, weights, nodes = make_nd_quadrature_rules(d, N)

        for i in range(multi_indices.shape[0]):
            computed_result = jnp.sum(weights * monomials(nodes, multi_indices[i]))
            expected_result = rms[i]
            npt.assert_almost_equal(computed_result, expected_result, decimal=12)

    def test_quadratic(self, d, N):
        """Test E[(X + z)^T A (X + z)] = tr(A cov) + (mean + z)^T A (mean + z)
        """
        mean, cov, _, _, weights, nodes = make_nd_quadrature_rules(d, N)

        zs = np.random.randn(d)
        A = np.random.randn(d, d)

        def f(xs):
            return jnp.dot(xs + zs, A @ (xs + zs))

        computed_result = jnp.sum(weights * jax.vmap(f, in_axes=[0])(nodes))
        expected_result = jnp.trace(A @ cov) + jnp.dot(mean + zs, A @ (mean + zs))
        npt.assert_allclose(computed_result, expected_result, rtol=1e-14)

    def test_mgf(self, d, N):
        """Test E[exp(z^T x)]
        """
        mean, cov, _, _, weights, nodes = make_nd_quadrature_rules(d, N)

        def f(xs, _zs):
            return jnp.exp(jnp.dot(_zs, xs))

        zs = np.random.randn(d)

        computed_result = jnp.sum(weights * jax.vmap(f, in_axes=[0, None])(nodes, zs))
        expected_result = jnp.exp(jnp.dot(zs, mean) + 0.5 * jnp.dot(zs, cov @ zs))
        npt.assert_allclose(computed_result, expected_result, rtol=1e-3)

    def test_invariance(self, d, N):
        """Quadrature using different moment modes should be the same.
        """
        key = jax.random.PRNGKey(666)
        _c = jax.random.normal(key, (d, d)) * 0.1
        cov = _c @ _c.T + jnp.eye(d)

        key, _ = jax.random.split(key)
        mean = jax.random.normal(key, (d,))

        multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        rms = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean, cov, multi_indices)
        cms = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(cov, multi_indices)
        scale = np.sqrt(np.diag(cov))
        scms = cms / np.array([np.prod(scale ** multi_index) for multi_index in multi_indices])

        weights, nodes = moment_quadrature_nd(rms, inds, sort_nodes=True)
        weights2, nodes2 = moment_quadrature_nd(cms, inds, mean, sort_nodes=True)
        weights3, nodes3 = moment_quadrature_nd(scms, inds, mean, scale, sort_nodes=True)

        npt.assert_allclose(nodes, nodes2, rtol=1e-10)
        npt.assert_allclose(nodes, nodes3, rtol=1e-10)

        if N % 2 == 0:
            # TODO: The weights are not equal when N is odd. THIS IS ODD!
            npt.assert_array_almost_equal(weights, weights2, decimal=10)
            npt.assert_array_almost_equal(weights, weights3, decimal=10)

        @partial(jax.vmap, in_axes=[0])
        def test_fn(x):
            return jnp.cos(x[0]) + x[1] * x[0]

        r1 = jnp.sum(weights * test_fn(nodes))
        r2 = jnp.sum(weights2 * test_fn(nodes2))
        r3 = jnp.sum(weights3 * test_fn(nodes3))
        npt.assert_allclose(r1, r2, rtol=1e-12)
        npt.assert_allclose(r1, r3, rtol=1e-12)

    def test_nd_uniform(self, d, N):
        """Test nd uniform moments.
        """
        bounds = [(-0.5, 0.5)] * d
        volume = np.prod([bound[1] - bound[0] for bound in bounds])
        npt.assert_equal(volume, 1.)

        multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
        inds = gram_and_hankel_indices_graded_lexico(N, d)

        rms = np.vectorize(moments_nd_uniform, signature='(d)->()', excluded=[0])(bounds, multi_indices)
        weights, nodes = moment_quadrature_nd(rms, inds)

        for i in range(multi_indices.shape[0]):
            computed_result = np.sum(weights * monomials(nodes, multi_indices[i]))
            expected_result = rms[i]
            npt.assert_almost_equal(computed_result, expected_result, decimal=15)

    def test_2d_polygon_uniform(self, d, N):
        """Test the moments of a polygon-uniform distribution.
        """
        if d != 2:
            """Only test d=2
            """
            return True
        else:
            multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
            inds = gram_and_hankel_indices_graded_lexico(N, d)

            # 2D polygon uniform distribution
            sym_x, sym_y = symbols('x, y')
            vertices = [(-0.5, -0.5), (0.1, 0.5), (0.5, -0.5)]
            polygon = Polygon(*vertices)
            normalising_constant = polytope_integrate(polygon, 1)

            rms = []
            for multi_index in multi_indices:
                expr = sym_x ** multi_index[0] * sym_y ** multi_index[1]
                rm = polytope_integrate(polygon, expr) / normalising_constant
                rms.append(float(rm.evalf()))

            rms = jnp.array(rms)
            weights, nodes = moment_quadrature_nd(rms, inds)

            for i in range(multi_indices.shape[0]):
                computed_result = np.sum(weights * monomials(nodes, multi_indices[i]))
                expected_result = rms[i]
                npt.assert_almost_equal(computed_result, expected_result, decimal=15)


def _assert_perm(a, b, func, *args, **kwargs):
    return func(np.sort(a, axis=0), np.sort(b, axis=0), *args, **kwargs)


@pytest.mark.parametrize('d', [1, 2])
@pytest.mark.parametrize('N', [2, 6])
@pytest.mark.parametrize('cov_type', ['any', 'diag'])
def test_vs_gauss_hermite(d, N, cov_type):
    np.random.seed(999)
    mean = jnp.asarray(np.random.randn(d))
    if cov_type == 'any':
        _c = np.random.randn(d, d)
        cov = _c @ _c.T
    else:
        cov = jnp.eye(d)

    sgps = SigmaPoints.gauss_hermite(d, order=N)

    def f(x):
        return jnp.sum(jnp.tanh(x), axis=-1)

    gh_nodes = sgps.gen_sigma_points(mean, jnp.linalg.cholesky(cov))
    gh_result = jnp.einsum('i,i', sgps.w, f(gh_nodes))

    multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
    inds = gram_and_hankel_indices_graded_lexico(N, d)

    rms = np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean, cov, multi_indices)
    cms = np.vectorize(central_moments_mvn_kan, signature='(d,d),(d)->()')(cov, multi_indices)

    weights, nodes = moment_quadrature_nd(rms, inds)
    weights2, nodes2 = moment_quadrature_nd(cms, inds, mean)

    npt.assert_array_almost_equal(weights, weights2, decimal=10)
    npt.assert_array_almost_equal(nodes, nodes2, decimal=8)

    gauss_result1 = jnp.einsum('i,i', weights, f(nodes))

    if cov_type == 'diag' and N == 2:
        _assert_perm(sgps.w, weights[np.abs(weights) > 1e-10], npt.assert_array_almost_equal, decimal=12)
        _assert_perm(gh_nodes, nodes[np.abs(weights) > 1e-10], npt.assert_array_almost_equal, decimal=12)
        npt.assert_array_almost_equal(gauss_result1, gh_result, decimal=15)

    if N == 2:
        npt.assert_allclose(gauss_result1, gh_result, rtol=2e-1)
    else:
        npt.assert_allclose(gauss_result1, gh_result, rtol=2e-4)
