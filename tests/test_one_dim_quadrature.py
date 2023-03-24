import math
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from mfs.one_dim.quadtures import moment_quadrature, taylor_quadrature
from mfs.one_dim.moments import raw_moment_of_normal, raw_to_central, raw_to_scaled
from jax.config import config
import pytest

np.random.seed(666)
config.update("jax_enable_x64", True)


def quadrature(f, ms, _mean=0., _scale=1.):
    ws, xs = moment_quadrature(ms, _mean, _scale)
    return jnp.einsum('i,i', f(xs), ws)


def gen_normal_raw_moments(_mean, _variance, _order):
    return jnp.array([raw_moment_of_normal(_mean, _variance, n) for n in range(_order + 1)])


def one_dim_gaussian_expectation_marginal(_mean, _variance):
    return 1 / (math.sqrt(2 * math.pi * (_variance + 3))) * jnp.exp(-(2 - _mean) ** 2 / (2 * (_variance + 3)))


def one_dim_gaussian_expectation_exp(_mean, _variance):
    return jnp.exp(_mean + _variance / 2)


def one_dim_gaussian_expectation_sine(_mean, _variance):
    return jnp.sin(_mean) * jnp.exp(-_variance / 2)


def one_dim_gaussian_expectation_poly(_mean, ms):
    return _mean + 2 * ms[2] + 1.1 * ms[3]


np.random.seed(666)
order = 15
mean, variance = 0.2, 1.1
scale = math.sqrt(variance)
rms = gen_normal_raw_moments(mean, variance, order)
cs = raw_to_central(rms)
ss = raw_to_scaled(rms)


class TestGaussQuadrature:

    def test_invariance(self):
        """Invariance with moment mode.
        """
        ws, xs = moment_quadrature(rms)
        for item in (moment_quadrature(cs, mean), moment_quadrature(ss, mean, scale)):
            _ws, _xs = item
            npt.assert_array_almost_equal(_ws, ws)
            npt.assert_array_almost_equal(_xs, xs)

    def test_gaussian_marginal(self):
        """Use the quadrature to do marginalisation.
        """
        def f(x):
            return 1 / (math.sqrt(2 * math.pi * 3)) * jnp.exp(-(x - 2) ** 2 / (2 * 3))

        computed_expectations = [quadrature(f, rms), quadrature(f, cs, mean), quadrature(f, ss, mean, scale)]
        theoretical_expectation = one_dim_gaussian_expectation_marginal(mean, variance)
        for computed_expectation in computed_expectations:
            npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-6)

    def test_gaussian_exp(self):

        computed_expectations = [quadrature(jnp.exp, rms), quadrature(jnp.exp, cs, mean),
                                 quadrature(jnp.exp, ss, mean, scale)]
        theoretical_expectation = one_dim_gaussian_expectation_exp(mean, variance)
        for computed_expectation in computed_expectations:
            npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-7)

    def test_gaussian_sine(self):

        computed_expectations = [quadrature(jnp.sin, rms), quadrature(jnp.sin, cs, mean),
                                 quadrature(jnp.sin, ss, mean, scale)]
        theoretical_expectation = one_dim_gaussian_expectation_sine(mean, variance)
        for computed_expectation in computed_expectations:
            npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-6)

    def test_gaussian_poly(self):
        """Should be exact up to the moment order.
        """
        ms = gen_normal_raw_moments(mean, variance, 3)
        cs = raw_to_central(ms)
        ss = cs / jnp.array([jnp.sqrt(variance) ** n for n in range(3 + 1)])

        def f(x):
            return x + 2 * x ** 2 + 1.1 * x ** 3

        computed_expectations = [quadrature(f, ms), quadrature(f, cs, mean), quadrature(f, ss, mean, scale)]
        theoretical_expectation = one_dim_gaussian_expectation_poly(mean, ms)
        for computed_expectation in computed_expectations:
            npt.assert_allclose(computed_expectation, theoretical_expectation)

    @pytest.mark.parametrize('_order', [3, 5, 7])
    def test_uniform_poly(self, _order):
        a, b = -2, 3.
        rms = jnp.array([1 / (k + 1) * sum([a ** i * b ** (k - i) for i in range(k + 1)]) for k in range(_order + 1)])
        coeffs = np.random.randn(_order)

        def f(x):
            return sum([coeffs[k] * x ** k for k in range(_order)])

        theoretical_expectation = sum([coeffs[k] / (k + 1) * (b ** (k + 1) - a ** (k + 1))
                                       for k in range(_order)]) / (b - a)
        computed_expectation = quadrature(f, rms, 0., 1.)
        npt.assert_allclose(computed_expectation, theoretical_expectation)


class TestTaylorQuadrature:

    def test_gaussian_marginal(self):
        order = 7

        mean, variance = 0.2, 1.1
        rms = gen_normal_raw_moments(mean, variance, order)
        cms = raw_to_central(rms)

        def f(x):
            return 1 / (math.sqrt(2 * math.pi * 3)) * jnp.exp(-(x - 2) ** 2 / (2 * 3))

        computed_expectation = taylor_quadrature(f, cms, mean, order=order)
        theoretical_expectation = one_dim_gaussian_expectation_marginal(mean, variance)
        npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-2)

    def test_gaussian_sine(self):
        order = 7

        mean, variance = 0.2, 1.1
        rms = gen_normal_raw_moments(mean, variance, order)
        cms = raw_to_central(rms)

        computed_expectation = taylor_quadrature(jnp.sin, cms, mean, order=order)
        theoretical_expectation = one_dim_gaussian_expectation_sine(mean, variance)
        npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-2)

    def test_gaussian_exp(self):
        order = 7

        mean, variance = 0.2, 1.1
        rms = gen_normal_raw_moments(mean, variance, order)
        cms = raw_to_central(rms)

        computed_expectation = taylor_quadrature(jnp.exp, cms, mean, order=order)
        theoretical_expectation = one_dim_gaussian_expectation_exp(mean, variance)
        npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-2)

    def test_gaussian_poly(self):
        order = 3

        mean, variance = 0.2, 1.1
        rms = gen_normal_raw_moments(mean, variance, order)
        cms = raw_to_central(rms)

        def f(x, arg):
            return x + 2 * (x - arg) ** 2 + 1.1 * (x - arg) ** 3

        computed_expectation = taylor_quadrature(f, cms, mean, order, mean)
        theoretical_expectation = one_dim_gaussian_expectation_poly(mean, cms)
        npt.assert_allclose(computed_expectation, theoretical_expectation, rtol=1e-6)
