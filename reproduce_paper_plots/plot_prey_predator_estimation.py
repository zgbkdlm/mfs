import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, \
    gram_and_hankel_indices_graded_lexico
from mfs.multi_dims.moments import sde_cond_moments_euler_maruyama, sde_cond_moments_tme_normal, sde_cond_moments_tme, \
    extract_mean, extract_cov, marginalise_moments
from mfs.multi_dims.filtering import moment_filter_nd_rms, moment_filter_nd_cms
from mfs.multi_dims.ss_models import prey_predator
from jax.config import config

config.update("jax_enable_x64", True)

d = 2
N = 5
transition = 'tme_2'
mode = 'central'

# Multi-indices
multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
inds = gram_and_hankel_indices_graded_lexico(N, d)

# Model
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(multi_indices)
rms0, cms0, mean0 = init_cond.rms, init_cond.cms, init_cond.mean

if 'euler' in transition:
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_euler_maruyama(drift, dispersion,
                                                                                        dt, multi_indices)
    indexing = 'index'
elif 'tme_normal' in transition:
    tme_order = int(transition.split('_')[-1])
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_tme_normal(drift, dispersion,
                                                                                    dt, tme_order, multi_indices)
    indexing = 'index'
else:
    tme_order = int(transition.split('_')[-1])
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_tme(drift, dispersion, dt, tme_order)
    indexing = 'multi-index'

# Create filter
if mode == 'raw':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_nd_rms((sde_cond_rms, indexing), measurement_cond_pmf, _ys,
                                    (jnp.asarray(multi_indices), inds), rms0)
elif mode == 'central':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_nd_cms((sde_cond_cms, indexing), state_cond_mean, measurement_cond_pmf, _ys,
                                    (jnp.asarray(multi_indices), inds), cms0, mean0)
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')

# Monte Carlo
key = jax.random.PRNGKey(999)

_, xs, ys = simulate(key)

# Moment filtering

if mode == 'raw':
    rmss, _ = moment_filter(ys)
    means = extract_mean(rmss, d)
    covs = extract_cov(rmss, d) - np.einsum('...i,...j->...ij', means, means)
elif mode == 'central':
    cmss, means, _ = moment_filter(ys)
    covs = extract_cov(cmss, d)
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharey='row')

axes[0].plot(ts, xs[:, 0], c='black', linestyle=':', linewidth=2, label='True trajectory')
axes[0].plot(ts, means[:, 0], c='black', linewidth=2, label='MF mean')
axes[0].fill_between(ts,
                     means[:, 0] - 1.96 * np.sqrt(covs[:, 0, 0]),
                     means[:, 0] + 1.96 * np.sqrt(covs[:, 0, 0]),
                     edgecolor='none',
                     facecolor='black',
                     alpha=0.15)
axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_xlabel('$t$')

axes[1].plot(ts, xs[:, 1], c='black', linestyle=':', linewidth=2, label='True trajectory')
axes[1].plot(ts, means[:, 1], c='black', linewidth=2, label='MF mean')
axes[1].fill_between(ts,
                     means[:, 1] - 1.96 * np.sqrt(covs[:, 1, 1]),
                     means[:, 1] + 1.96 * np.sqrt(covs[:, 1, 1]),
                     edgecolor='none',
                     facecolor='black',
                     alpha=0.15)
axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_xlabel('$t$')

plt.tight_layout(pad=0.1)
plt.show()
