"""
Common utilities.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from mfs.typings import JArray, FloatScalar, JFloat, Array
from typing import Sequence, Callable, Tuple, Union, NamedTuple


def gamma(x):
    return jnp.exp(jax.lax.lgamma(x))


def factorial(n):
    return gamma(n + 1.)


def binom(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


class GaussianSum1D(NamedTuple):
    """Unidimensional Gaussian-sum distribution.
    """
    means: JArray
    variances: JArray
    weights: JArray
    mean: JFloat
    variance: JFloat
    rms: JArray
    cms: JArray
    scms: JArray

    def pdf(self, xs):
        pdfs = jax.scipy.stats.norm.pdf(jnp.atleast_1d(xs)[:, None], self.means, jnp.sqrt(self.variances))
        return jnp.sum(pdfs * self.weights[None, :], axis=1)

    def sampler(self, key, n):
        cs = jax.random.choice(key, self.means.shape[0], (n,), p=self.weights)
        key, _ = jax.random.split(key)
        return self.means[cs] + jnp.sqrt(self.variances[cs]) * jax.random.normal(key, (n,))

    @classmethod
    def new(cls, means: JArray, variances: JArray, weights: JArray, N: int = 2):
        # Avoid circular import
        from mfs.one_dim.moments import raw_moment_of_normal

        centre = jnp.sum(means * weights)
        rms = jnp.array([sum([raw_moment_of_normal(mean, variance, p) * weight for mean, variance, weight in
                              zip(means, variances, weights)]) for p in range(2 * N)])
        cms = jnp.array([sum([raw_moment_of_normal(mean - centre, variance, p) * weight for mean, variance, weight in
                              zip(means, variances, weights)]) for p in range(2 * N)])
        variance = cms[2]
        scms = cms / jnp.sqrt(variance) ** jnp.arange(2 * N)
        return cls(means=means, variances=variances, weights=weights,
                   mean=centre, variance=variance,
                   rms=rms, cms=cms, scms=scms)


class GaussianSumND(NamedTuple):
    """Multidimensional Gaussian-sum distribution.
    """
    d: int
    means: JArray  # (n, d)
    covs: JArray  # (n, d, d)
    weights: JArray  # (n, )
    mean: JArray
    cov: JArray
    rms: JArray
    cms: JArray

    def pdf(self, x):
        pdfs = [jax.scipy.stats.multivariate_normal.pdf(x, mean, cov) * weight for mean, cov, weight in
                zip(self.means, self.covs, self.weights)]
        return jnp.sum(jnp.array(pdfs))

    def logpdf(self, x):
        """Log pdf using logsumexp
        """
        log_pdfs = jnp.array(
            [jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov) for mean, cov in zip(self.means, self.covs)])
        return jax.scipy.special.logsumexp(log_pdfs, b=self.weights)

    def sampler(self, key, nsamples):
        cs = jax.random.choice(key, self.means.shape[0], (nsamples,), p=self.weights)
        key, _ = jax.random.split(key)
        return self.means[cs, :] + jnp.einsum('...ij,...j->...i', jnp.linalg.cholesky(self.covs[cs, :, :]),
                                              jax.random.normal(key, (nsamples, self.d)))

    @classmethod
    def new(cls, means: JArray, covs: JArray, weights: JArray, multi_indices):
        from mfs.multi_dims.moments import raw_moments_mvn_kan

        d = means.shape[1]

        centre = jnp.sum(means * weights[:, None], axis=0)
        cov = sum(
            [weight * (cov + jnp.outer(mean, mean)) for mean, cov, weight in zip(means, covs, weights)]) - jnp.outer(
            centre, centre)
        l = [np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean, cov, multi_indices) * weight for
             mean, cov, weight in zip(means, covs, weights)]
        rms = jnp.sum(jnp.vstack(l), axis=0)

        l = [
            np.vectorize(raw_moments_mvn_kan, signature='(d),(d,d),(d)->()')(mean - centre, cov, multi_indices) * weight
            for mean, cov, weight in zip(means, covs, weights)]
        cms = jnp.sum(jnp.vstack(l), axis=0)
        return cls(d=d, means=means, covs=covs, weights=weights, mean=centre, cov=cov, rms=rms, cms=cms)


def discretise_lti_sde(A: JArray, B: JArray, dt: float) -> Tuple[JArray, JArray]:
    """Analytically discretise linear time-invariant SDEs of the form

        dX(t) = A X(t) dt + B dW(t),

    to

        X(t_k) = F(t_k) X(t_{k-1}) + Q(t_k),   Q(t_k) ~ N(0, Cov[X(t_k) | X(t_{k-1})]),

    for any t_k > t_{k-1}. Remark that E[X(t_k) | X(t_{k-1})] = F(t_k) X(t_{k-1}), and that the ODEs that
    govern m(t) and V(t) in the lecture note.

    Arguments
    ---------
    A : JArray (d, d)
        Drift matrix.
    B : JArray (d, w)
        Dispersion matrix.
    dt : float
        Time interval t_{k} - t_{k-1}.

    Returns
    -------
    JArray (d, d), JArray (d, d)
        The transition matrix and the covariance.

    References
    ----------
    Axelsson and Gustafsson. Discrete-time solutions to the continuous-time differential Lyapunov equation with
    applications to Kalman filtering, 2015.

    Applied stochastic differential equations, 2019, pp. 83
    """
    d = A.shape[0]

    F = jax.scipy.linalg.expm(A * dt)
    phi = jnp.vstack([jnp.hstack([A, B @ B.T]), jnp.hstack([jnp.zeros_like(A), -A.T])])
    AB = jax.scipy.linalg.expm(phi * dt) @ jnp.vstack([jnp.zeros_like(A), jnp.eye(d)])
    cov = AB[0:d, :] @ F.T
    return F, cov


def vmap_list_of_funcs(funcs: Sequence[Callable]) -> Callable:
    """vmap a list of functions.

    Suppose that funcs = (f1, f2, f3), then this returns a function z such that z(x) = jnp.array([f1(x), f2(x), f3(x)])
    executed in vmap.

    References
    ----------
    https://github.com/google/jax/issues/673#issuecomment-894955037.
    """

    def select_func(ind: jnp.ndarray, x):
        return jax.lax.switch(ind, funcs, x)

    def vmap_func(x):
        return jax.vmap(select_func, in_axes=(0, None))(jnp.arange(len(funcs)), x)

    return vmap_func


def simulate_sde(m_and_cov: Callable[[JArray, FloatScalar], Tuple[JArray, JArray]],
                 x0: Union[FloatScalar, JArray],
                 dt: FloatScalar,
                 T: int,
                 key: JArray,
                 diagonal_cov: bool = False,
                 integration_steps: int = 1) -> JArray:
    """Simulate an (approximate) SDE trajectory by small Gaussian increments on uniform time grids.

    Parameters
    ----------
    m_and_cov : (d, ), () -> (d, ), (d, d)
        SDE conditional mean and covariance.
    x0 : float or JArray (d, )
        The initial value.
    dt : FloatScalar
        Uniform time interval.
    T : int
        Number of time steps.
    key : JArray
        Random key.
    diagonal_cov : bool, default=False
        Indicate whether the conditional SDE covariance is diagonal.
    integration_steps : int, default=1
        Perform :code:`integration_steps` integration steps at each time interval for lower SDE discretisation error.

    Returns
    -------
    JArray (T, d)
        The SDE trajectory evaluated at the times.
    """
    x0 = jnp.atleast_1d(x0)
    d = x0.shape[0]

    ddt = dt / integration_steps

    key, _ = jax.random.split(key)
    normal_increments = jax.random.normal(key=key, shape=(T, integration_steps, d))

    def integration_body(carry, elem):
        x = carry
        normal_inc = elem

        m, cov = m_and_cov(x, ddt)

        if diagonal_cov:
            x = m + jnp.sqrt(cov) @ normal_inc
        else:
            x = m + jax.lax.linalg.cholesky(cov) @ normal_inc
        return x, None

    def scan_body(carry, elem):
        x = carry
        normal_inc = elem

        x, _ = jax.lax.scan(integration_body, x, normal_inc)
        return x, x

    _, traj = jax.lax.scan(scan_body, x0, normal_increments)
    return traj


def partial_bell(n: int, k: int, xs: Union[Array, Sequence[float]]) -> FloatScalar:
    r"""The partial/incomplete Bell polynomial. See https://en.wikipedia.org/wiki/Bell_polynomials.

    Computed by using the recurrence relation

    .. math::

        B_{n, k}(x_1, x_2, \ldots, x_{n-k+1}) = \sum^{n-k+1}_{i=1} \binom{n-1}{i-1} \, x_i \, B_{n-i, k-1}(x_1, x_2, \ldots)

    Parameters
    ----------
    n : int
        See the equation above.
    k : int
        See the equation above.
    xs : Array-like (n - k + 1, )
        See the equation above. Can be a tuple of float, numpy array, or jax array. It is jittable only when xs is
        a JAX array.

    Returns
    -------
    FloatScalar
        The Bell polynomial evaluation.
    """
    if n == 0 and k == 0:
        return 1.
    elif (n >= 1 and k == 0) or (n == 0 and k >= 1):
        return 0.
    else:
        terms = [math.comb(n - 1, i - 1) * xs[i - 1] * partial_bell(n - i, k - 1, xs) for i in range(1, n - k + 1 + 1)]
        return sum(terms)


def complete_bell(n: int, xs: Union[Array, Sequence[float]]) -> FloatScalar:
    r"""The complete Bell polynomial.

    .. math::

        B_n(x_1, x_2, \ldots, x_n) = \sum^n_{k=1} B_{n, k}(x_1, x_2, \ldots, x_{n-k+1}).

    Parameters
    ----------
    n : int
        See the equation above.
    xs : Array-like (n - k + 1, )
        See the equation above. Can be a tuple of float, numpy array, or jax array. It is jittable only when xs is
        a jax array.

    Returns
    -------
    FloatScalar
        The complete Bell polynomial evaluation.
    """
    if n == 0:
        return 1.
    else:
        return sum([partial_bell(n, k, xs) for k in range(1, n + 1)])


def hermite_probabilist(n: int, x: FloatScalar) -> FloatScalar:
    r"""Probabilist's Hermite polynomial, n-th order.

    Computed by using the three-term recurrence

    .. math::

        H_{n + 1}(x) = x \, H_n(x) - n \, H_{n - 1}(x)

    Parameters
    ----------
    n : int
        Order
    x : FloatScalar
        x.

    Returns
    -------
    FloatScalar
        H_n(x).
    """
    if n == 0:
        return 1.
    elif n == 1:
        return x
    else:
        return x * hermite_probabilist(n - 1, x) - (n - 1) * hermite_probabilist(n - 2, x)


def lanczos(a: JArray, v0: JArray, m: int) -> Tuple[JArray, JArray, JArray]:
    r"""Lanczos algorithm.

    .. math::

        A = V \, T \, V^T

    Parameters
    ----------
    a : JArray (n, n)
        A symmetric matrix of size n. This function does not deal with complex matrices.
    v0 : JArray (n, )
        An arbitrary vector with Euclidean norm 1, for example, v0 = [1, 0, 0, ...].
    m : int
        Number of Lanczos iterations (must >= 1 and <= n).

    Returns
    -------
    JArray (n, m), JArray (m, ), JArray (m - 1, )
        V, the diagonal of T, and the 1-off diagonal of T.

    References
    ----------
    Gene H. Golub and Charles F. Van Load. Matrix computations. 2013. The Johns Hopkins University Press, 4th edition.

    https://en.wikipedia.org/wiki/Lanczos_algorithm.

    Notes
    -----
    Let (\lambda, u) be a pair of eigenvalue and eigenvector of T, then we can approximately use (\lambda, V u) as
    the eigenvalue and eigenvector of A. When m = n, this is precise.
    """

    def scan_body(carry, _):
        _v, w = carry

        beta = jnp.sqrt(jnp.sum(w ** 2))
        v = w / beta
        wp = a @ v
        alpha = jnp.dot(wp, v)
        w = wp - alpha * v - beta * _v

        return (v, w), (v, alpha, beta)

    wp0 = a @ v0
    alpha0 = jnp.dot(wp0, v0)
    w0 = wp0 - alpha0 * v0

    _, (vs, alphas, betas) = jax.lax.scan(scan_body, (v0, w0), jnp.arange(m - 1))
    return jnp.vstack([v0, vs]).T, jnp.hstack([alpha0, alphas]), betas


def lanczos_ritz(a: JArray, v0: JArray, m: int,
                 eigh_tridiagonal: bool = False, sort_eigenvalues: bool = True) -> Tuple[JArray, JArray]:
    r"""Compute the Ritz vector and values from Lanczos algorithm.

    Parameters
    ----------
    a : JArray (n, n)
        A symmetric matrix of size n. This function does not deal with complex matrices.
    v0 : JArray (n, )
        An arbitrary vector.
    m : int
        Number of Lanczos iterations (must >= 1 and <= n).
    eigh_tridiagonal : bool, default=False
        Flag this argument to True to enable fast eigendecompositions of tridiagonal matrix.
    sort_eigenvalues : bool, default=True
        Sort the eigenvalues.

    Returns
    -------
    JArray (n, m), JArray (m, )
        Ritz vectors and values.

    Notes
    -----
    Unlike the function `lanczos`, the input vector `v0` here needs not to be normalised. This function will normalise
    it by itself.

    This function is not used in this version but for future works.
    """
    if eigh_tridiagonal:
        raise NotImplementedError('At the time of writing down this implementation, '
                                  'jax.scipy.linalg.eigh_tridiagonal was not fully supported.')
    norm = jnp.linalg.norm(v0)
    vs, alphas, betas = lanczos(a, v0 / norm, m)
    T = jnp.diag(alphas) + jnp.diag(betas, k=-1) + jnp.diag(betas, k=1)
    pre_eigenvectors, eigenvalues = jax.lax.linalg.eigh(T, sort_eigenvalues=sort_eigenvalues)
    return jnp.einsum('ik,kj,j->ij', vs, pre_eigenvectors, pre_eigenvectors[0, :] * norm), eigenvalues


def posterior_cramer_rao(state_trajectories: JArray,
                         measurements: JArray,
                         j0: JArray,
                         logpdf_transition: Callable[[JArray, JArray], FloatScalar],
                         logpdf_likelihood: Callable[[JArray, JArray], FloatScalar]) -> JArray:
    """Compute the posterior Cramér--Rao lower bound at given times by Monte Carlo.

    Parameters
    ----------
    state_trajectories : jnp.ndarray (T + 1, N, dx)
        Trajectories of the state. T, N, and dx are the number of times, number of MC samples, and state dimension,
        respectively. Note that one should have the initial samples in the beginning.
    measurements : jnp.ndarray (T, N, dy)
        Measurements. dy represents the dimension of the measurement variable.
    j0 : jnp.ndarray (dx, dx)
        The initial -E[H_X log p(x0)].
    logpdf_transition : (dx, ), (dx, ) -> float
        Log p(x_k | x_{k-1})
    logpdf_likelihood : (dy, ), (dx, ) -> float
        Log p(y_k | x_k)

    Returns
    -------
    jnp.ndarray (T, dx, dx)
        J, the inverse of the PCRLB lower bound matrices.

    Notes
    -----
    There are a number of posterior Cramér--Rao lower bounds in the literature (see, Frutsche et al., 2014). We here use
    the one by Tichavsky et al., 1998, see, also, Challa et al., 2011. Although this one is not the tightest, it is
    easiest to compute (with Monte Carlo approximation). Other bounds need the filtering distributions to integrate out
    the latent variables which are not exactly tractable.

    References
    ----------
    Subhash Challa, Mark Morelande, Darko Musicki, and Robin Evans. Fundamentals of object tracking. Cambridge
    University Press, 2011, pp. 53.

    Peter Tichavsky, Carlos Muravchik, and Arye Nehorai. Posterior Cramér--Rao bounds for discrete-time nonlinear
    filtering. IEEE Transactions on Signal Processing, 1998.

    Carsteb Frutsche, Emre Ozkan, Lennart Svensson, and Fredrik Gustafsson. A fresh look at Bayesian Cramér--Rao
    bounds for discrete-time nonlinear filtering. In 17th International Conference on Information Fusion, 2014.
    """
    htt_logpdf_transition = jax.vmap(jax.hessian(logpdf_transition, argnums=0), in_axes=[0, 0])
    hts_logpdf_transition = jax.vmap(jax.jacfwd(jax.jacrev(logpdf_transition, argnums=1), argnums=0), in_axes=[0, 0])
    hss_logpdf_transition = jax.vmap(jax.hessian(logpdf_transition, argnums=1), in_axes=[0, 0])
    htt_logpdf_likelihood = jax.vmap(jax.hessian(logpdf_likelihood, argnums=1), in_axes=[0, 0])

    def scan_body(carry, elem):
        j = carry
        yt_mcs, xt_mcs, xs_mcs = elem

        d11 = -jnp.mean(hss_logpdf_transition(xt_mcs, xs_mcs), axis=0)
        d12 = -jnp.mean(hts_logpdf_transition(xt_mcs, xs_mcs), axis=0)
        d22 = -jnp.mean(htt_logpdf_transition(xt_mcs, xs_mcs) + htt_logpdf_likelihood(yt_mcs, xt_mcs), axis=0)

        j = d22 - d12.T @ jnp.linalg.solve(j + d11, d12)
        return j, j

    _, js = jax.lax.scan(scan_body, j0, (measurements, state_trajectories[1:], state_trajectories[:-1]))
    return js


def ldl(mat: JArray) -> Tuple[JArray, JArray]:
    """LDL decomposition.

    mat = L D L^T, where L is the lower triangular matrix.

    Parameters
    ----------
    mat: JArray (n, n)
        A symmetric matrix.

    Returns
    -------
    JArray (n, n), JArray (n, )
        The lower triangular matrix and diagonal vector, respectively.

    Notes
    -----
    The current JAX version does not natively support LDL. This implementation is only for proof-of-concept, which is
    not performant at all. The XLA team should implement LDL in the XLA level for the best performance.
    """
    n = mat.shape[0]
    l = jnp.eye(n).at[1:, 0].set(mat[1:, 0] / mat[0, 0])
    d = jnp.ones((n,)) * mat[0, 0]
    for j in range(1, n):
        v = l[j, :j] * d[:j]
        _d = mat[j, j] - jnp.dot(l[j, :j], v)
        d = d.at[j].set(_d)
        l = l.at[j + 1:, j].set((mat[j + 1:, j] - l[j + 1:, :j] @ v) / _d)
    return l, d


def ldl_chol(mat: JArray, eps: float = None) -> JArray:
    """PD matrix completion.

    Notes
    -----
    The epsilon should be chosen according to e.g., Cheng1998.

    https://nhigham.com/2020/12/22/what-is-a-modified-cholesky-factorization/
    """
    if eps is None:
        eps = 1e-8 * jnp.linalg.norm(mat, 'fro')
    l, d = ldl(mat)
    return jnp.einsum('ij,j->ij', l, jnp.where(d < 0, eps, jnp.sqrt(d)))
