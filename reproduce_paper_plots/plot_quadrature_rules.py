import scipy
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mfs.one_dim.quadtures import moment_quadrature
from mfs.one_dim.moments import raw_moment_of_normal, raw_to_central
from mfs.utils import GaussianSum1D
from jax.config import config

config.update("jax_enable_x64", True)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

fig, axes = plt.subplots(ncols=3, sharey='row', figsize=(17, 5))

# Gaussian mixture
means = jnp.array([-2., 2.])
variances = jnp.array([0.5, 1.])
weights = jnp.array([0.7, 0.3])

N = 11
order = 2 * N - 1
rms = jnp.array([sum([raw_moment_of_normal(m, v, p) * w for m, v, w in zip(means, variances, weights)])
                 for p in range(order + 1)])
cms = raw_to_central(rms)

gs = GaussianSum1D.new(means=means, variances=variances, weights=weights)

ws, xs = moment_quadrature(cms, jnp.sum(means * weights))

ss = jnp.linspace(-7, 7, 200)
axes[0].plot(ss, gs.pdf(ss), c='black', linewidth=2)
# axes[0].scatter(xs, jnp.zeros(ws.shape), s=ws * 1e3, c='black', alpha=0.5, edgecolors='none')
axes[0].scatter(xs, ws, s=100, c='black', alpha=0.5, edgecolors='none')
axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$p(x)$')
axes[0].set_title(r'$0.7 \, \mathcal{N}(-2, 0.5) + 0.3 \, \mathcal{N}(2, 1)$')

# Uniform
a, b = -2., 2.
rms = jnp.array([1 / (k + 1) * sum([a ** i * b ** (k - i) for i in range(k + 1)]) for k in range(order + 1)])

ws, xs = moment_quadrature(rms)

ss = jnp.linspace(-3, 3, 3)
axes[1].hlines(1 / (b - a), a, b, colors='black', linewidth=2)
# axes[1].scatter(xs, jnp.zeros(ws.shape), s=ws * 1e3, c='black', alpha=0.5, edgecolors='none')
axes[1].scatter(xs, ws, s=100, c='black', alpha=0.5, edgecolors='none')
axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_xlabel('$x$')
axes[1].set_title(r'$U(-2, 2)$')

# Gamma
k, theta = 2., 1.
rms = jnp.array([theta ** n * math.gamma(k + n) / math.gamma(k) for n in range(order + 1)])

ws, xs = moment_quadrature(rms)

ss = jnp.linspace(0, 35, 200)
axes[2].plot(ss, scipy.stats.gamma.pdf(ss, a=k, scale=theta), c='black', linewidth=2,
             label='PDF')
# axes[2].scatter(xs, jnp.zeros(ws.shape), s=ws * 1e3, c='black', alpha=0.5, edgecolors='none',
#                 label='Quadrature nodes and weights')
axes[2].scatter(xs, ws, s=100, c='black', alpha=0.5, edgecolors='none',
                label='Quadrature rules')
axes[2].grid(linestyle='--', alpha=0.3, which='both')
axes[2].set_xlabel('$x$')
axes[2].set_title(r'$\mathrm{Gamma}(2, 1)$')
axes[2].legend()

plt.tight_layout(pad=0.1)
plt.savefig('1d_rules.pdf')
plt.show()
