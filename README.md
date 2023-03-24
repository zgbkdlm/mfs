# Stochastic filtering with moment representation
[![UnitTest](https://github.com/zgbkdlm/mfs/actions/workflows/unittest.yml/badge.svg)](https://github.com/zgbkdlm/mfs/actions/workflows/unittest.yml)

Implementation of the moment filter introduced in the paper "Stochastic filtering with moment representation". Please 
cite as follows to use the implementation.

```bibtex
@article{
    author = {Zheng Zhao and Juha Sarmarvuori}, 
    title = {Stochastic filtering with moment representation},
    journal = {arXiv preprint arXiv:placeholder},
    year = {2023},
}
```

Please feel free to find the preprint at https://zz.zabemon.com/resources/pdfs/mfs.pdf or https://arxiv.org/abs/placeholder.

# Moment filtering
Consider a model

```math
\begin{equation}
    \begin{split}
        \mathrm{d} X(t) &= a(X(t)) \mathrm{d} t + b(X(t)) \mathrm{d} W(t),\\
        X_0 &\sim \mathbb{P}_{X_0}, \\
        Y_k \mid X_k &\sim p_{Y_k \mid X_k},
    \end{split}
\end{equation}
```

or $X$ being a discrete-time process $X_k \mid X_{k-1} \sim \mathbb{P}\_{X_k \mid X_{k-1}}$. This paper proposes a routine 
to recursively compute the moments

```math
\begin{equation}
    \begin{split}
        M_k^N &\coloneqq \lbrace m_{k, 0},  m_{k, 1},  m_{k, 2}, \ldots, m_{k, 2 \, N - 1} \rbrace, \\
        m_{k, n} &\coloneqq \mathbb{E}[{X_k^n \mid Y_{1:k}}] \coloneqq \int x^n \mathrm{d} \mathbb{P}_{X_k \mid Y_{1:k}}(x),
    \end{split}
\end{equation}
```

for $k=1,2,\ldots$ and any $N$, which approximately represent the filtering distribution $\mathbb{P}\_{X_k \mid Y_{1:k}}$.

Under mild system conditions, the filter converges to the true solution **in moments and distribution** as $N\to\infty$. Moreover, the filter also gives an asymptotically exact (log) likelihood for parameter estimation, although it's biased. The implementation in JAX allows this likelihood be differentiable in the parameter.

![](./docs/source/figs/banner.gif "Check ./examples/benes_bernoulli.ipynb")

# Installation

The implementation is based on JAX. Depending on your computer platform (e.g., CPU/GPU/TPU), the installation of JAX can be different. Hence, please first refer to this official [guidance](https://github.com/google/jax#installation) to install JAX by yourself.

After you have JAX installed, then do

1. `git clone git@github.com:zgbkdlm/mfs.git` or `git clone https://github.com/zgbkdlm/mfs.git`.
2. `cd mfs`
3. `pip install -r requirements.txt`
4. `pip install -r testing_requirements.txt`
5. `python setup.py install` or `python setup.py develop` for the editable model. If `setup.py` is deprecated, then do `python -m pip install .` instead.

# Example

You can find a few examples in `./examples` to help you use get started with the moment filter.

A sketch of using raw moments for 1D filtering is given as follows. 

```python
import jax
from mfs.one_dim.filtering import moment_filter_rms

# Define your model here
def sde_cond_rms(x, n):
    """The transition moment E[X_k^n | X_{k-1} = x].
    """
    return ...

def pdf_y_cond_x(y, x):
    """The measurement PDF p(y | x).
    """
    return ...

# Your data
ys = ...

# Initial raw moments
rms0 = ...

# JIT moment filter
@jax.jit
def moment_filter(_ys):
    return moment_filter_rms(sde_cond_rms, pdf_y_cond_x, rms0, _ys)

# rmss are the filtering raw moments, and nell is the negative log-likelihood
rmss, nell = moment_filter(ys)
```

# Reproduce the results

To *exactly* reproduce the figures/tables/numbers in the paper, please refer to the instructions in `./reproduce_paper_plots`, and also the instructions in `./dardel`.

# Other useful contents

During the development time of this work, I have also experimented a bunch of side-implementations in JAX, which are related/unrelated to this moment filter. 
I would be glad if you find them useful for your projects:

- A bunch of commonly used filters and smoothers, such as extended Kalman filter, sigma-points filters, and particle filters (`mfs.classical_filters_smoothers`).
- **Brute-force filter** (`mfs.classical_filters_smoothers.brute_force`). This can handily compute the true filtering solution for 1D state up to machine precision. You can use this as a benchmark to gauge your method.
- The Kan--Magnus method for efficiently computing Gaussian moments (`mfs.multi_dims.moments`). 
- Graded lexicographical ordering (`mfs.multi_dims.multi_indices`).
- Gram--Charlier series (`mfs.one_dim.pdf_approximations`).
- Saddle point approximation (`mfs.one_dim.pdf_approximations`).
- Posterior Cramér--Rao lower bound for filtering (`mfs.utils`).
- Partial and complete Bell polynomials (`mfs.utils`).
- Legendre polynomial expansion (`mfs.one_dim.pdf_approximations`).
- Lánczos algorithm (`mfs.utils`).

# For MATLAB and Julia users
In the coming days, we will upload some demonstrations written in MATLAB and Julia under the folders `./matlab` and `./julia`, respectively. Please note that these implementations are for proof-of-concept only, and that they do not reproduce the results in the paper.

# License
The GNU General Public License v3 or later. See `./LICENSE`.

# Contact
Zheng Zhao, Uppsala University, firstname.lastname@it.uu.se, https://zz.zabemon.com.

Juha Sarmavuori, Aalto University, firstname.lastname@aalto.fi.
