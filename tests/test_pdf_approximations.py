import math
import jax
import jax.scipy
import jax.numpy as jnp
import numpy.testing as npt
from mfs.one_dim.pdf_approximations import truncated_cumulant_generating_function, saddle_point, gram_charlier, \
    legendre_poly_expansion, inverse_fourier
from mfs.one_dim.moments import raw_moment_of_normal, raw_to_central, raw_to_scaled, sms_to_cumulants
from jax.config import config

config.update("jax_enable_x64", True)


class TestSaddlePointApproximation:

    def test_cgf_mgf_with_normal(self):
        mean, variance = jnp.array([0.3, 1.2])
        rms = jnp.array([raw_moment_of_normal(mean, variance, p) for p in range(10)])
        cms = raw_to_central(rms)
        sms = raw_to_scaled(rms)

        def true_cgf(_z):
            return mean * _z + 0.5 * variance * _z ** 2

        z = jnp.array(1.)
        theoretical_cfg = true_cgf(z)

        npt.assert_allclose(truncated_cumulant_generating_function(z, rms), theoretical_cfg, rtol=1e-3)
        npt.assert_allclose(truncated_cumulant_generating_function(z, cms, mean), theoretical_cfg, rtol=1e-3)
        npt.assert_allclose(truncated_cumulant_generating_function(z, sms, mean, jnp.sqrt(variance)),
                            theoretical_cfg, rtol=1e-3)

    def test_saddle_point_approx_normal(self):
        mean, variance = jnp.array([0.3, 1.2])
        rms = jnp.array([raw_moment_of_normal(mean, variance, p) for p in range(15)])
        sms = raw_to_scaled(rms)

        pdf = saddle_point(sms, mean, jnp.sqrt(variance))
        xs = jnp.linspace(-5, 5, 100)

        # NOT TESTED!
        # npt.assert_allclose(pdf(xs), jax.scipy.stats.norm.pdf(xs, mean, jnp.sqrt(variance)))


class TestGramCharlier:

    def test_gram_charlier_normal(self):
        mean, variance = jnp.array([1.1, 1.2])
        rms = jnp.array([raw_moment_of_normal(mean, variance, p) for p in range(4)])
        sms = raw_to_scaled(rms)
        cumulants = sms_to_cumulants(sms, mean, jnp.sqrt(variance))

        xs = jnp.linspace(-5, 5, 100)

        computed_pdf = jax.jit(gram_charlier(cumulants))(xs)
        theoretical_pdf = jax.scipy.stats.norm.pdf(xs, mean, jnp.sqrt(variance))
        npt.assert_allclose(computed_pdf, theoretical_pdf, rtol=1e-12)

    def test_two_normal(self):
        key = jax.random.PRNGKey(666)
        samples_1 = -1. + 0.5 * jax.random.normal(key, (100000, ))

        key, _ = jax.random.split(key)
        samples_2 = 1. + 0.8 * jax.random.normal(key, (100000, ))

        samples = jnp.concatenate([samples_1, samples_2])
        mean = jnp.mean(samples)
        scale = jnp.std(samples)
        rms = jnp.array([jnp.mean(samples ** n) for n in range(10)])
        sms = raw_to_scaled(rms)
        cumulants = sms_to_cumulants(sms, mean, scale)

        xs = jnp.linspace(-5, 5, 100)
        computed_pdf = gram_charlier(cumulants)

        # plt.plot(xs, computed_pdf(xs))
        # plt.hist(samples, density=True, bins=100)
        # plt.show()


class TestLegendre:

    def test_legendre_expansion_normal(self):
        key = jax.random.PRNGKey(666)
        samples = jnp.tanh(jax.random.normal(key, (100000, )))

        rms = jnp.array([jnp.mean(samples ** n) for n in range(15)])

        a, b = -1., 1.
        pdf = jax.jit(legendre_poly_expansion(rms, a, b))

        xs = jnp.linspace(a, b, 100)

        # plt.plot(xs, pdf(xs))
        # plt.hist(samples, density=True, bins=100)
        # plt.show()


class TestInverseFourier:

    def test_inverse_fourier(self):
        mean, variance = 0.1, 0.2

        zs = jnp.linspace(-20, 20, 1000)
        cfs = jnp.exp(1.j * zs * mean - 0.5 * variance * zs ** 2)

        xs = jnp.linspace(-2, 2, 100)
        pdf_estimated = jax.vmap(inverse_fourier, in_axes=[0, None, None])(xs, cfs, zs)

        npt.assert_allclose(jax.scipy.stats.norm.pdf(xs, mean, math.sqrt(variance)), pdf_estimated, rtol=1e-10)
