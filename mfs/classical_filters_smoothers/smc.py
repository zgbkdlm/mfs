"""
Filters and smoothers based on sequential Monte Carlo, for instance, particle filters and smoothers.
"""
import jax
import jax.numpy as jnp
from mfs.classical_filters_smoothers.resampling import continuous_resampling
from mfs.typings import JArray, FloatScalar
from typing import Callable, Tuple


def bootstrap_filter(transition_sampler: Callable[[JArray, JArray], JArray],
                     measurement_cond_pdf: Callable[[JArray, FloatScalar], JArray],
                     ys: JArray,
                     init_sampler: Callable[[JArray, int], JArray],
                     key: JArray,
                     nsamples: int,
                     resampling: Callable[[JArray, JArray], JArray],
                     conti_resampling: bool = False) -> Tuple[JArray, FloatScalar]:
    """Bootstrap particle filter.

    Parameters
    ----------
    transition_sampler : (n, dx), key -> (n, dx)
        Draw n new samples based on the previous samples. In the bootstrap filter, we use the transition density.
    measurement_cond_pdf : (dy, ), (n, dx) -> (n, )
        p(y | x). The first function argument is for y. The second argument is for x, which accepts an array of samples
        and output an array of evaluations.
    ys : JArray (T, dy)
        Measurements.
    init_sampler : key, int -> (n, dx)
        A function that makes samples from the initial distribution.
    key : JArray
        PRNGKey.
    nsamples : int
        Number of samples/particles n.
    resampling : (n, ), key -> (n, )
        Resample method.
    conti_resampling : bool, default=False
        Whether use the continuous resampling for giving the correct gradients.

    Returns
    -------
    JArray (T, n, dx), FloatScalar
        The filtering samples and the negative log likelihood.
    """

    def scan_body(carry, elem):
        samples, nell = carry
        y, _key = elem

        samples = transition_sampler(samples, _key)

        weights = measurement_cond_pdf(y, samples)
        nell -= jnp.log(jnp.mean(weights))
        weights = weights / weights.sum()

        _key, _ = jax.random.split(_key)
        if conti_resampling:
            samples = continuous_resampling(samples, weights, nsamples, _key)
        else:
            samples = samples[resampling(weights, _key), ...]

        return (samples, nell), samples

    init_samples = init_sampler(key, nsamples)
    keys = jax.random.split(key, num=ys.shape[0])

    (*_, nell_ys), solution_samples = jax.lax.scan(scan_body, (init_samples, 0.), (ys, keys))
    return solution_samples, nell_ys


def particle_filter(proposal_sampler: Callable[[JArray, JArray, JArray], JArray],
                    proposal_density: Callable[[JArray, JArray, JArray], JArray],
                    transition_density: Callable[[JArray, JArray], JArray],
                    measurement_cond_pdf: Callable[[JArray, FloatScalar], JArray],
                    ys: JArray,
                    init_sampler: Callable[[JArray, int], JArray],
                    key: JArray,
                    nsamples: int,
                    resampling: Callable[[JArray, JArray], JArray]) -> JArray:
    """The standard particle filter.

    Parameters
    ----------
    proposal_sampler : (n, dx), dy, key -> (n, dx)
        Draw n new samples based on the previous samples according to the variance-optimal
        distribution :math:`p(x_k | x_{k-1}, y_k)`.
    proposal_density : (n, dx), (n, dx), dy -> (n, )
        The variance-optimal density function :math:`p(x_k | x_{k-1}, y_k)`.
    transition_density : (n, dx), (n, dx) -> (n, )
        The transition density.
    measurement_cond_pdf : (dy, ), (n, dx) -> (n, )
        p(y | x). The first function argument is for y. The second argument is for x, which accepts an array of samples
        and output an array of evaluations.
    ys : JArray (T, dy)
        Measurements.
    init_sampler : key, int -> (n, dx)
        A function that makes samples from the initial distribution.
    key : JArray
        PRNGKey.
    nsamples : int
        Number of samples/particles n.
    resampling : (n, ), key -> (n, )
        Resample method.

    Returns
    -------
    JArray (T, n, dx)
    """

    def scan_body(carry, elem):
        ancestors = carry
        y, _key = elem

        samples = proposal_sampler(ancestors, y, _key)
        weights = measurement_cond_pdf(y, samples) * transition_density(samples, ancestors) \
                  / proposal_density(samples, ancestors, y)
        weights = weights / weights.sum()

        _key, _ = jax.random.split(_key)
        samples = samples[resampling(weights, _key), ...]
        return samples, samples

    init_samples = init_sampler(key, nsamples)
    keys = jax.random.split(key, num=ys.shape[0])
    return jax.lax.scan(scan_body, init_samples, (ys, keys))[1]
