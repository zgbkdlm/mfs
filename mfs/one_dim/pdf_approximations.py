"""
Approximate probability density based on moments.
"""
import math
import jax
import jax.numpy as jnp
import jax.scipy.stats
from mfs.utils import complete_bell, hermite_probabilist
from mfs.typings import JArray, JFloat, FloatScalar
from functools import partial
from typing import Callable, Union, Sequence


# def saddle_point(sms: JArray,
#                  mean: FloatScalar,
#                  scale: FloatScalar) -> Callable:
#     """https://en.wikipedia.org/wiki/Saddlepoint_approximation_method
#     """
#
#     def cgf(z):
#         return truncated_cumulant_generating_function(z, sms, mean, scale)
#
#     d_cgf = jax.grad(cgf)
#     dd_cgf = jax.grad(d_cgf)
#
#     @partial(jax.vmap, in_axes=[0])
#     def pdf(x: DeviceFloat) -> DeviceFloat:
#         def saddle_equation(s):
#             s = jnp.squeeze(s)
#             return d_cgf(s) - x
#
#         opt_solver = jaxopt.GaussNewton(residual_fun=saddle_equation)
#         init_s = jnp.array([(x - mean) / scale ** 2])
#         s, _ = opt_solver.run(init_params=init_s)
#         s = jnp.squeeze(s)
#         return jnp.exp(cgf(s) - s * x) / jnp.sqrt(2 * math.pi * dd_cgf(s))
#
#     return pdf


def gram_charlier(cumulants: JArray) -> Callable[[JArray], JArray]:
    """Gram--Charlier A series (with Normal distribution as the base/reference distribution).

    Parameters
    ----------
    cumulants : JArray (2 n - 1, )
        Cumulants k_1, ..., k_{2 n - 1}. You can use the function :code:`moments.sms_to_cumulants` to convert scaled
        central moments to the cumulants.

    Returns
    -------
    (..., ) -> (..., )
        The PDF function approximated by the Gram--Charlier A series.

    Notes
    -----
    The compilation of this function might take a while for high-order cumulants.
    """
    order = cumulants.shape[0]
    mean = cumulants[0]
    variance = cumulants[1]

    bell_input = jnp.concatenate([jnp.array([0., 0.]), cumulants[2:]])

    def base_func(x: JFloat) -> JFloat:
        return 1 / jnp.sqrt(2 * math.pi * variance) * jnp.exp(-(x - mean) ** 2 / 2 / variance)

    @partial(jax.vmap, in_axes=[0])
    def pdf(x):
        h = ((x - mean) / jnp.sqrt(variance))
        z = [complete_bell(j, bell_input[:j]) / (math.factorial(j) * variance ** (j / 2)) * hermite_probabilist(j, h)
             for j in range(order + 1)]
        return base_func(x) * sum(z)

    return pdf


def edgeworth():
    """Edgeworth series
    """


def legendre_poly_expansion(rms: JArray, a: FloatScalar = -1., b: FloatScalar = 1.) -> Callable:
    """Legendre polynomial expansion for densities in a compact interval.

    Parameters
    ----------
    rms : JArray (2 n, )
        Raw moments.
    a : FloatScalar, default=-1
        Left interval.
    b : FloatScalar, default=-1
        Right interval.

    Returns
    -------
    (..., ) -> (..., )
        A callable PDF.
    """
    num_moments = rms.shape[0]

    def poly(k: int, placeholder: Union[Sequence[FloatScalar], JArray]) -> FloatScalar:
        return sum([(-1) ** i * 2 ** (-k) * math.factorial(2 * k - 2 * i)
                    / (math.factorial(i) * math.factorial(k - i) * math.factorial(k - 2 * i))
                    * placeholder[k - 2 * i]
                    for i in range(math.floor(k / 2) + 1)])

    def legendre(k: int, x: FloatScalar) -> FloatScalar:
        return poly(k, jnp.array([x ** i for i in range(k + 1)]))

    def basis_coeff(k: int) -> FloatScalar:
        return (2 * k + 1) / 2 * poly(k, rms)

    @partial(jax.vmap, in_axes=[0])
    def pdf(x):
        return 2 / (b - a) * sum([basis_coeff(k) * legendre(k, (2 * x - (a + b)) / (b - a))
                                  for k in range(num_moments)])

    return pdf


def truncated_cumulant_generating_function(z: JFloat,
                                           ms: JArray,
                                           mean: FloatScalar = 0.,
                                           scale: FloatScalar = 1.) -> JFloat:
    """Truncated cumulant-generating function :math:`K(z)`.

    Depending on if the arguments :code:`mean` and :code:`scale` are given, the argument :code:`ms` can be
    raw/central/scaled. See the document below.

    Parameters
    ----------
    z : JFloat
    ms
    mean
    scale

    Returns
    -------

    """
    num_moments = ms.shape[0]
    smgf_coeffs = jnp.array([(z * scale) ** n / math.factorial(n) for n in range(num_moments)])
    smgf = jnp.einsum('i,i', smgf_coeffs, ms)
    return z * mean + jnp.log(smgf)


def saddle_point(sms: JArray,
                 mean: FloatScalar,
                 scale: FloatScalar) -> Callable:
    """https://en.wikipedia.org/wiki/Saddlepoint_approximation_method.

    This implementation applies to polynomial-truncated cumulant-generating functions.
    """
    num_moments = sms.shape[0]
    sms_poly = jnp.flip(sms / jnp.array([math.factorial(n) for n in range(num_moments)]))

    def cgf(z):
        return z * mean + jnp.log(jnp.polyval(sms_poly, z * scale))

    d_cgf = jax.grad(cgf)
    dd_cgf = jax.grad(d_cgf)

    def find_nearest_real_root(roots, p):
        infed_roots = jnp.where(jnp.abs(roots.imag) < 1e-8, roots, jnp.inf)
        return jnp.real(roots[jnp.argmin(jnp.abs(infed_roots - (p + 0.j)))])

    @partial(jax.vmap, in_axes=[0])
    def pdf(x: JFloat) -> JFloat:
        saddle_equation_poly = jnp.polyadd((mean - x) * sms_poly, scale * jnp.polyder(sms_poly))
        s = find_nearest_real_root(saddle_equation_poly, jnp.array([(x - mean) / scale ** 2])) / scale
        return jnp.exp(cgf(s) - s * x) / jnp.sqrt(2 * math.pi * dd_cgf(s))

    return pdf


def inverse_fourier(x: FloatScalar, cfs: JArray, zs: JArray) -> JFloat:
    r"""Compute the probability density function by inverse Fourier transform of the characteristic function.

    .. math::

        p(x) = \int \exp(- \mathrm{i} \, x \, z) \varphi(z) dz

    Parameters
    ----------
    x : FloatScalar
        At where the density function is evaluated.
    cfs : JArray[Complex] (m, )
        An array of the values of the characteristic function evaluated at `zs`.
    zs : JArray (m, )
        At where the characteristic function is evaluated.
    Returns
    -------
    JFloat[Real]
        The density value :math:`p(x)`.
    """
    return jnp.real(jnp.trapz(jnp.exp(-1.j * x * zs) * cfs, zs)) / (2 * math.pi)
