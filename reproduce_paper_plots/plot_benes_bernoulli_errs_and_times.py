"""
Plot the filtering errors for the Benes--Bernoulli model.
"""
import numpy as np
import matplotlib.pyplot as plt
from mfs.one_dim.ss_models import benes_bernoulli

dt, T, ts, *_ = benes_bernoulli(2)


def rm_divergent(arr):
    """This computes the number of divergences.
    """
    mask = np.any(np.isnan(arr), axis=1)
    return arr[~mask, :], np.count_nonzero(mask)


# Unpack all the errors
def unpack(b: int = 2, m: int = 2000, mode: str = 'central', _t: str = '_normal', N: int = 15):
    all_errs = np.load(f'../dardel/results/benes_bernoulli_errs/errs_b_{b}_m_{m}_{mode}{_t}_N_{N}.npz')

    _, true_ndivs = rm_divergent(all_errs['true_trajs_vs_true_means_abs'])

    true_means_vs_mf_means_abs, ndiv_mf = rm_divergent(all_errs['true_means_vs_mf_means_abs'])
    true_means_vs_ghf_means_abs, ndiv_ghf = rm_divergent(all_errs['true_means_vs_ghf_means_abs'])
    true_means_vs_pf_means_abs, ndiv_pf = rm_divergent(all_errs['true_means_vs_pf_means_abs'])

    true_cfs_vs_mf_sup, _ = rm_divergent(all_errs['true_cfs_vs_mf_sup'])
    true_cfs_vs_ghf_sup, _ = rm_divergent(all_errs['true_cfs_vs_ghf_sup'])
    true_cfs_vs_pf_sup, _ = rm_divergent(all_errs['true_cfs_vs_pf_sup'])

    print(f'Bad simulations: {true_ndivs}')
    print(f'MF divergences: {ndiv_mf - true_ndivs}')
    print(f'GHF divergences: {ndiv_ghf - true_ndivs}')
    print(f'PF divergences: {ndiv_pf - true_ndivs}')
    return true_cfs_vs_mf_sup, true_cfs_vs_ghf_sup, true_cfs_vs_pf_sup


true_cfs_vs_mf_sup, true_cfs_vs_ghf_sup, true_cfs_vs_pf_sup = unpack()

# Load all the time results
Ns = np.arange(2, 16)
num_mcs = 1000
nparticles = 10000

times_mf = np.zeros((Ns.shape[0], num_mcs))
for k, N in enumerate(Ns):
    times_mf[k] = np.load(f'../dardel/results/times/mf_raw_N_{N}.npz')['elapsed_times']

times_ghf = np.load(f'../dardel/results/times/ghf_11.npz')['elapsed_times']
times_ghf = np.tile(times_ghf, (Ns.shape[0], 1))

times_pf = np.load(f'../dardel/results/times/pf_{nparticles}.npz')['elapsed_times']
times_pf = np.tile(times_pf, (Ns.shape[0], 1))

# Plotting setting
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 18})

# Plot the errs in t
plt.plot(ts, np.nanmean(true_cfs_vs_mf_sup, axis=0), c='black', linewidth=2, label='Moment filter ($N=15$)')
plt.plot(ts, np.nanmean(true_cfs_vs_ghf_sup, axis=0), c='black', linestyle=':', linewidth=2,
         label='Gauss--Hermite filter')
plt.plot(ts, np.nanmean(true_cfs_vs_pf_sup, axis=0), c='black', linestyle='--', linewidth=2, label='Particle filter')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlabel('$t$')
plt.ylabel(r'$\sup_z \bigl\lvert \phi_t(z) - \widetilde{\phi}_t(z) \bigr\rvert$')
plt.yscale('log')
plt.title('Errors of the characteristic functions in $t$')

plt.legend()
plt.tight_layout(pad=0.1)
plt.savefig('./errs_cf_in_time.pdf')
plt.show()

## Plot the errors and times
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Plot the errors in N
errs_mf_cfs_means = np.zeros((Ns.shape[0],))
# errs_mf_cfs_stds = np.zeros((Ns.shape[0],))
errs_mf_cfs_quantiles = np.zeros((2, Ns.shape[0]))
for i, N in enumerate(Ns):
    true_cfs_vs_mf_sup, _, _ = unpack(N=N)
    errs_mf_cfs_means[i] = np.mean(np.mean(true_cfs_vs_mf_sup, axis=1))
    # errs_mf_cfs_stds[i] = np.std(np.mean(true_cfs_vs_mf_sup, axis=1))
    errs_mf_cfs_quantiles[0, i] = np.quantile(np.mean(true_cfs_vs_mf_sup, axis=1), 0.025)
    errs_mf_cfs_quantiles[1, i] = np.quantile(np.mean(true_cfs_vs_mf_sup, axis=1), 0.975)

axes[0].plot(Ns, errs_mf_cfs_means, c='black', linewidth=2, label='Moment filter')
axes[0].fill_between(np.arange(2, 16),
                     errs_mf_cfs_quantiles[0],
                     errs_mf_cfs_quantiles[1],
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[0].hlines(np.mean(true_cfs_vs_ghf_sup), 2, 15, colors='black', linestyle=':', linewidth=2,
               label='Gauss--Hermite filter')
axes[0].fill_between(np.arange(2, 16),
                     np.quantile(true_cfs_vs_ghf_sup, 0.025),
                     np.quantile(true_cfs_vs_ghf_sup, 0.975),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[0].hlines(np.mean(true_cfs_vs_pf_sup), 2, 15, colors='black', linestyle='--', linewidth=2, label='Particle filter')
axes[0].fill_between(np.arange(2, 16),
                     np.quantile(true_cfs_vs_pf_sup, 0.025),
                     np.quantile(true_cfs_vs_pf_sup, 0.975),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_xticks(Ns)
axes[0].set_xlabel('$N$')
axes[0].set_ylabel(r'$\sum_{k=1}^{100}\sup_{z} \lvert \varphi_k(z) - \widehat{\varphi}_k(z) \rvert / 100$')
axes[0].set_yscale('log')
axes[0].legend(loc='lower left', fontsize=20)

# Plot the time
axes[1].plot(Ns, np.mean(times_mf, axis=1), c='black', linewidth=2, label='Moment filter')
axes[1].fill_between(Ns,
                     np.mean(times_mf, axis=1) - 2 * np.std(times_mf, axis=1),
                     np.mean(times_mf, axis=1) + 2 * np.std(times_mf, axis=1),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

axes[1].plot(Ns, np.mean(times_ghf, axis=1), c='black', linewidth=2, linestyle=':', label='Gauss--Hermite filter')
axes[1].fill_between(Ns,
                     np.mean(times_ghf, axis=1) - 2 * np.std(times_ghf, axis=1),
                     np.mean(times_ghf, axis=1) + 2 * np.std(times_ghf, axis=1),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

axes[1].plot(Ns, np.mean(times_pf, axis=1), c='black', linewidth=2, linestyle='--', label='Particle filter')
axes[1].fill_between(Ns,
                     np.mean(times_pf, axis=1) - 2 * np.std(times_pf, axis=1),
                     np.mean(times_pf, axis=1) + 2 * np.std(times_pf, axis=1),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_yscale('log')
axes[1].set_xlabel('$N$')
axes[1].set_xticks(Ns, Ns.astype('int64'))
axes[1].set_ylabel('Computational time (second)')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0.16)
plt.savefig('./errs_cf_in_N_and_times.pdf')
plt.show()
