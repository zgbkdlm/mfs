import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
import matplotlib.pyplot as plt
from mfs.classical_filters_smoothers.brute_force import brute_force_filter
from mfs.one_dim.pdf_approximations import inverse_fourier
from mfs.one_dim.ss_models import benes_bernoulli
from mfs.one_dim.filtering import moment_filter_cms, moment_filter_rms, moment_filter_scms
from mfs.one_dim.moments import sde_cond_moments_tme_normal, sde_cond_moments_tme, characteristic_fn, \
    characteristic_from_pdf
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from mfs.classical_filters_smoothers.gfs import sgp_filter
from mfs.classical_filters_smoothers.smc import bootstrap_filter
from mfs.classical_filters_smoothers.resampling import stratified
from jax.config import config

config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(1991)

# Settings
N = 15
tme_order = 3
dt, T, ts, init_cond, drift, dispersion, logistic, measurement_cond_pmf, simulate_trajectory = benes_bernoulli(N)
use_normal_transition = True
mode = 'central'
gh_order = 11
nparticles = 10000


# Ground-truth filter
@jax.jit
def ground_truth_filter(_spatial_grids, _ys):
    return brute_force_filter(drift, dispersion, measurement_cond_pmf,
                              init_cond.pdf(_spatial_grids), _spatial_grids, _ys, dt,
                              integration_steps=100, pred_method='chapman-tme-3')


# Moment filter
if use_normal_transition:
    sde_cond_rms, sde_cond_cms, sde_cond_scms, state_cond_mean, state_cond_mean_var = sde_cond_moments_tme_normal(
        drift,
        dispersion,
        dt,
        tme_order,
        N)
else:
    sde_cond_rms, sde_cond_cms, sde_cond_scms, state_cond_mean, state_cond_mean_var = sde_cond_moments_tme(
        drift, dispersion,
        dt, tme_order)

if mode == 'raw':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_rms(sde_cond_rms, measurement_cond_pmf,
                                 init_cond.rms, _ys)
elif mode == 'central':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_cms(sde_cond_cms, state_cond_mean, measurement_cond_pmf,
                                 init_cond.cms, init_cond.mean, _ys)
elif mode == 'scaled':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_scms(sde_cond_scms, state_cond_mean_var, measurement_cond_pmf,
                                  init_cond.scms, init_cond.mean, jnp.sqrt(init_cond.variance), _ys)
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')


# Gauss--Hermite
def state_cond_m_cov(x, _dt):
    return tme.mean_and_cov(jnp.atleast_1d(x), _dt, drift, dispersion, order=tme_order)


def measurement_cond_m_cov(x):
    p = logistic(x)
    return jnp.atleast_1d(p), jnp.atleast_2d(p * (1 - p))


sgps = SigmaPoints.gauss_hermite(d=1, order=gh_order)


@jax.jit
def ghf(_ys):
    return sgp_filter(state_cond_m_cov, measurement_cond_m_cov, sgps,
                      jnp.atleast_1d(init_cond.mean), jnp.atleast_2d(init_cond.variance), dt,
                      _ys[:, None], const_measurement_cov=False)


# The particle filter
def proposal_sampler(x, _key):
    ms, covs = jax.vmap(state_cond_m_cov, in_axes=[0, None])(x, dt)
    return jnp.squeeze(ms) + jnp.squeeze(jnp.sqrt(covs)) * jax.random.normal(_key, (nparticles,))


@jax.jit
def pf(_ys, _key):
    return bootstrap_filter(proposal_sampler, measurement_cond_pmf, _ys, init_cond.sampler, _key,
                            nparticles, stratified)[0]


# Simulate a trajectory and measurements
key_x0, key_xs, key_ys = jax.random.split(key, 3)
x0 = init_cond.sampler(key_x0, 1)[0]
xs = simulate_trajectory(x0, key_xs)
ys = jax.random.bernoulli(key_ys, logistic(xs), (T,))

if __name__ == '__main__':
    # Moment filter
    cmss, means, _ = moment_filter(ys)

    # Ground-truth filter
    spatial_lb = jnp.min(means - 5 * jnp.sqrt(cmss[:, 2]))
    spatial_ub = jnp.max(means + 5 * jnp.sqrt(cmss[:, 2]))
    spatial_grids = jnp.linspace(spatial_lb, spatial_ub, 2000)
    true_pdfs = ground_truth_filter(spatial_grids, ys)

    # Gauss-Hermite
    ghf_mfs, ghf_vfs, _ = ghf(ys)
    ghf_mfs, ghf_vfs = ghf_mfs[:, 0], ghf_vfs[:, 0, 0]

    # # Particle filter
    key_pf, _ = jax.random.split(key_ys)
    samples = pf(ys, key_pf)

    # Grids
    zs = jnp.linspace(-9, 9, 2000)

    # Plot setting
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': "serif",
        'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
        'font.size': 18})

    # Plot 3D
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_box_aspect(aspect=(2, 1, 1))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"].update({'linestyle': '--', 'alpha': 0.3})

    # ax.plot(ts, xs, zs=0, zdir='z', c='black', linewidth=2, linestyle='--')

    for k in range(0, T, 20):
        ax.plot(spatial_grids, true_pdfs[k, :], zs=ts[k], zdir='x', c='black', linewidth=2)

        estimated_cfs = jax.jit(jax.vmap(characteristic_fn, in_axes=[0, None, None]))(zs, cmss[k], means[k])
        estimated_pdfs = jax.jit(jax.vmap(inverse_fourier, in_axes=[0, None, None]))(spatial_grids, estimated_cfs,
                                                                                     zs)
        ax.plot(spatial_grids, estimated_pdfs, zs=ts[k], zdir='x', linestyle='--', linewidth=2, c='tab:purple',
                marker='s', markevery=400, alpha=0.5)

        hist, bin_edges = np.histogram(samples[k], bins=20, density=True)
        ax.bar(bin_edges[1:], hist, width=bin_edges[1] - bin_edges[0], zs=ts[k], zdir='x', alpha=0.2, color='black')

    ax.plot(spatial_grids, true_pdfs[T, :], zs=ts[-1], zdir='x', c='black', linewidth=2, label='True PDF')
    estimated_cfs = jax.jit(jax.vmap(characteristic_fn, in_axes=[0, None, None]))(zs, cmss[T], means[T])
    estimated_pdfs = jax.jit(jax.vmap(inverse_fourier, in_axes=[0, None, None]))(spatial_grids, estimated_cfs, zs)
    ax.plot(spatial_grids, estimated_pdfs, zs=ts[-1], zdir='x', linestyle='--', linewidth=2, c='tab:purple',
            marker='s', markevery=400, alpha=0.5, label='Moment filter PDF')
    hist, bin_edges = np.histogram(samples[T], bins=20, density=True)
    ax.bar(bin_edges[1:], hist, width=bin_edges[1] - bin_edges[0], zs=ts[T], zdir='x', alpha=0.2, color='black',
           label='Particle filter histogram')

    ax.set_xlabel('$t$')
    ax.set_ylabel('$X(t)$')
    ax.legend(loc='upper right', fontsize=18)
    # ax.dist = 8
    ax.set_box_aspect((2, 2, 1), zoom=1.2)

    # Plot 2D characteristic function
    ax2 = fig.add_subplot(1, 2, 2)

    k = 80
    true_cfs = jax.jit(jax.vmap(characteristic_from_pdf, in_axes=[0, None, None]))(zs, true_pdfs[k], spatial_grids)
    estimated_cfs = jax.jit(jax.vmap(characteristic_fn, in_axes=[0, None, None]))(zs, cmss[k], means[k])
    cfs_ghf = np.exp(1.j * zs * ghf_mfs[k] - 0.5 * zs ** 2 * ghf_vfs[k])
    cfs_pf = np.mean(np.exp(1.j * zs[None, :] * samples[k, :, None]), axis=0)

    ax2.plot(np.real(true_cfs), np.imag(true_cfs), linewidth=2, c='black', label=r'True $\varphi_t(z)$')
    ax2.plot(np.real(estimated_cfs), np.imag(estimated_cfs), linewidth=2, linestyle='--', c='tab:purple', alpha=0.7,
             marker='s', markevery=200, label=r'Moment filter $\widehat{\varphi}_t(z)$')
    ax2.plot(np.real(cfs_ghf), np.imag(cfs_ghf), linewidth=1, linestyle=':', c='black', alpha=0.7, marker='.',
             markevery=200, label=r'Gauss--Hermite $\widehat{\varphi}_t(z)$')
    ax2.plot(np.real(cfs_pf), np.imag(cfs_pf), linewidth=1, linestyle='--', c='black', alpha=0.2, marker='x',
             markevery=200, label=r'Particle filter $\widehat{\varphi}_t(z)$')
    ax2.grid(linestyle='--', alpha=0.3, which='both')
    ax2.legend(loc='lower right', fontsize=19)
    ax2.set_xlabel(r'$\mathrm{Im}(\varphi_t(z))$')
    ax2.set_ylabel(r'$\mathrm{Re}(\varphi_t(z))$')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    axins = ax2.inset_axes([0.5, 0.65, 0.3, 0.3])
    axins.plot(np.real(true_cfs), np.imag(true_cfs), linewidth=2, c='black')
    axins.plot(np.real(estimated_cfs), np.imag(estimated_cfs), linewidth=1, linestyle='--', c='tab:purple', alpha=0.7,
               marker='s', markevery=200)
    axins.plot(np.real(cfs_ghf), np.imag(cfs_ghf), linewidth=1, linestyle=':', c='black', alpha=0.7, marker='.',
               markevery=200)
    axins.plot(np.real(cfs_pf), np.imag(cfs_pf), linewidth=1, linestyle='--', c='black', alpha=0.2, marker='x',
               markevery=200)

    axins.set_xlim(-0.03, 0.01)
    axins.set_ylim(-0.08, 0.08)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax2.indicate_inset_zoom(axins, edgecolor="black")

    # Mark begin and end

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('benes_bernoulli_demo.pdf')
    plt.show()
