"""
Demonstrate the quadrature rules for a few 2D distributions.
"""
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sympy.core import symbols
from sympy.integrals.intpoly import Polygon, polytope_integrate
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, \
    gram_and_hankel_indices_graded_lexico
from mfs.multi_dims.quadratures import moment_quadrature_nd
from mfs.multi_dims.moments import moments_nd_uniform
from mfs.utils import GaussianSumND
from jax.config import config

np.random.seed(666)
config.update("jax_enable_x64", True)

# Plot setting
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 19})

marker_size = 2.e2

fig, axes = plt.subplots(nrows=3, ncols=3, sharey='row', sharex='col', figsize=(15, 11))

# Dimension
d = 2

# Plot for N = ...
for i, N in zip([0, 1, 2], [2, 4, 6]):

    multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=2 * N - 1)
    inds = gram_and_hankel_indices_graded_lexico(N, d)

    # 2D Gaussian sum
    means = jnp.array([[1., 1.],
                       [-1., -1.],
                       [-0.5, 0.5]])
    covs = jnp.array([[[0.2, 0.1],
                       [0.1, 1.]],
                      [[2., 0.2],
                       [0.2, 0.3]],
                      [[0.5, 0.],
                       [0., 1.]]])
    gs_weights = jnp.array([0.4, 0.4, 0.2])
    gs = GaussianSumND.new(means, covs, gs_weights, multi_indices)

    gs_pdf = jax.vmap(jax.vmap(gs.pdf, in_axes=[0]), in_axes=[0])

    weights, nodes = moment_quadrature_nd(gs.cms, inds, gs.mean)
    weights = np.asarray(weights / np.max(np.abs(weights)))

    pos_weights, pos_nodes = weights[weights >= 0.], nodes[weights >= 0.]
    neg_weights, neg_nodes = weights[weights < 0.], nodes[weights < 0.]

    x1s, x2s = np.mgrid[-4:3:0.01, -4:5:0.01]
    axes[i][0].contour(x1s, x2s, gs_pdf(np.dstack([x1s, x2s])), levels=10, linewidths=2, cmap=plt.cm.binary)
    axes[i][0].scatter(pos_nodes[:, 0], pos_nodes[:, 1], pos_weights * marker_size,
                       c='black', alpha=0.7, edgecolors='none', label='Positive weights')
    axes[i][0].scatter(neg_nodes[:, 0], neg_nodes[:, 1], -neg_weights * marker_size,
                       marker='x', c='black', alpha=0.7, label='Negative weights')
    axes[i][0].grid(linestyle='--', alpha=0.3, which='both')
    axes[0][0].legend(loc='upper left', fontsize=18)
    axes[0][0].set_title('Gaussian sum')
    axes[i][0].set_ylabel(f'$N = {N}$')

    # 2D uniform distribution
    bounds = [(-4., 4.), (-3., 4.)]
    volume = np.prod([bound[1] - bound[0] for bound in bounds])

    rms = np.vectorize(moments_nd_uniform, signature='(d)->()', excluded=[0])(bounds, multi_indices)
    weights, nodes = moment_quadrature_nd(rms, inds)
    weights = np.asarray(weights / np.max(np.abs(weights)))

    pos_weights, pos_nodes = weights[weights > 0.], nodes[weights > 0.]
    neg_weights, neg_nodes = weights[weights < 0.], nodes[weights < 0.]
    zero_weights, zero_nodes = weights[np.abs(weights) < 1e-8], nodes[np.abs(weights) < 1e-8]

    axes[i][1].fill([bounds[0][0], bounds[0][0], bounds[0][1], bounds[0][1]],
                    [bounds[1][0], bounds[1][1], bounds[1][1], bounds[1][0]],
                    fill=False, linewidth=2, c='black')
    axes[i][1].scatter(pos_nodes[:, 0], pos_nodes[:, 1], pos_weights * marker_size,
                       c='black', alpha=0.7, edgecolors='none')
    axes[i][1].scatter(neg_nodes[:, 0], neg_nodes[:, 1], -neg_weights * marker_size, marker='x', c='black', alpha=0.7)
    axes[i][1].scatter(zero_nodes[:, 0], zero_nodes[:, 1], s=20, marker='v',
                       facecolors='none', edgecolors='black', alpha=0.2)
    axes[i][1].grid(linestyle='--', alpha=0.3, which='both')
    axes[0][1].set_title('Uniform')

    # 2D polygon uniform distribution
    sym_x, sym_y = symbols('x, y')
    vertices = [(-4, -3), (0, 0), (-4, 4), (5, 1), (1, -2)]
    polygon = Polygon(*vertices)
    area = polytope_integrate(polygon, 1)

    rms = []
    for multi_index in multi_indices:
        expr = sym_x ** multi_index[0] * sym_y ** multi_index[1]
        rm = polytope_integrate(polygon, expr) / area
        rms.append(float(rm.evalf()))

    rms = jnp.array(rms)
    weights, nodes = moment_quadrature_nd(rms, inds)
    weights = np.asarray(weights / np.max(np.abs(weights)))

    pos_weights, pos_nodes = weights[weights >= 0.], nodes[weights >= 0.]
    neg_weights, neg_nodes = weights[weights < 0.], nodes[weights < 0.]

    _vertices_x_matplotlib = [vertex[0] for vertex in vertices]
    _vertices_y_matplotlib = [vertex[1] for vertex in vertices]
    axes[i][2].fill(_vertices_x_matplotlib, _vertices_y_matplotlib,
                    fill=False, linewidth=2, c='black')
    axes[i][2].scatter(pos_nodes[:, 0], pos_nodes[:, 1], pos_weights * marker_size,
                       c='black', alpha=0.7, edgecolors='none')
    axes[i][2].scatter(neg_nodes[:, 0], neg_nodes[:, 1], -neg_weights * marker_size, marker='x', c='black', alpha=0.7)
    axes[i][2].grid(linestyle='--', alpha=0.3, which='both')
    axes[0][2].set_title('Uniform-on-polygon')

plt.tight_layout(pad=0.1)
plt.savefig('2d_rules.pdf')
plt.show()
