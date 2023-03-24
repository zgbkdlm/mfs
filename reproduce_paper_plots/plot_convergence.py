import numpy as np
import matplotlib.pyplot as plt

dt = 1e-1
T = 100
ts = np.linspace(dt, dt * T, T)

path = '../dardel/results/convergence/'
mode = 'central'

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5), sharey='all', sharex='col')


def set_violins(_violins):
    for body in _violins['bodies']:
        body.set_facecolor('black')
        body.set_edgecolor('none')
        body.set_alpha(0.15)

    for part in ['cbars', 'cmins', 'cmaxes']:
        _violins[part].set_edgecolor('black')
        _violins[part].set_linestyle('--')
        _violins[part].set_linewidth(2)
        _violins[part].set_alpha(0.3)


axes[0, 0].set_ylabel(r'Errors of means')
axes[1, 0].set_ylabel(r'Errors of variances')

for i, N in enumerate([3, 7, 11, 15]):
    results = np.load(path + f'mf_{mode}_N_{N}.npz')
    errs_means = results['errs_means']
    errs_variances = results['errs_variances']
    errs_kls = np.abs(results['errs_kls'])

    # Plot mean errs
    axes[0, i].plot(ts, np.mean(errs_means, axis=0), c='black', linewidth=2, label=f'$N={N}$')
    violins = axes[0, i].violinplot(errs_means[:, 19:-1:20], positions=ts[19:-1:20], vert=True, widths=1,
                                    showmedians=False, showextrema=True, points=200)
    set_violins(violins)

    axes[0, i].set_yscale('log')
    axes[0, i].grid(linestyle='--', alpha=0.3, which='both')
    axes[0, i].set_title(f'$N = {N}$')

    # Plot variance errs
    axes[1, i].plot(ts, np.mean(errs_variances, axis=0), c='black', linewidth=2, label=f'$N={N}$')
    violins = axes[1, i].violinplot(errs_variances[:, 19:-1:20], positions=ts[19:-1:20], vert=True, widths=1,
                                    showmedians=False, showextrema=True, points=200)
    set_violins(violins)

    axes[1, i].set_yscale('log')
    axes[1, i].grid(linestyle='--', alpha=0.3, which='both')
    axes[1, i].set_xticks([0, 2, 4, 6, 8, 10], labels=[0, 2, 4, 6, 8, 10])
    axes[1, i].set_xlabel('$t$')

# Plot PF for comparison
results = np.load(path + 'pf.npz')
errs_means_pf = results['errs_means']
errs_variances_pf = results['errs_variances']
errs_kls = results['errs_kls']

axes[0, 1].plot(ts, np.mean(errs_means_pf, axis=0), c='black', linewidth=2, linestyle=':', label='PF')
axes[1, 1].plot(ts, np.mean(errs_variances_pf, axis=0), c='black', linewidth=2, linestyle=':', label='PF')

plt.tight_layout(pad=0.1)
plt.savefig('convergence_in_time.pdf')
plt.show()

# Plot convergence speed in N
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), sharey='all', sharex='col')

vals_means = np.zeros((10000, 14))
vals_vars = np.zeros((10000, 14))
for N in range(2, 16):
    results = np.load(path + f'mf_{mode}_N_{N}.npz')
    errs_means = results['errs_means']
    errs_variances = results['errs_variances']

    vals_means[:, N - 2] = np.mean(errs_means, axis=1)
    vals_vars[:, N - 2] = np.mean(errs_variances, axis=1)

axes[0].plot(np.arange(2, 16), np.mean(vals_means, axis=0), color='black', linewidth=2, label='Moment filter')
axes[0].fill_between(np.arange(2, 16),
                     np.mean(vals_means, axis=0) - 2 * np.std(vals_means, axis=0),
                     np.mean(vals_means, axis=0) + 2 * np.std(vals_means, axis=0),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[0].plot([2, 15], np.tile(np.mean(errs_means_pf), (2, )), color='black', linewidth=2, linestyle=':',
             label='Particle filter')
axes[0].fill_between([2, 15],
                     np.tile(np.mean(errs_means_pf), (2, )) - 2 * np.tile(np.std(np.mean(errs_means_pf, axis=1)), (2, )),
                     np.tile(np.mean(errs_means_pf), (2, )) + 2 * np.tile(np.std(np.mean(errs_means_pf, axis=1)), (2, )),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_xticks(np.arange(2, 16))
axes[0].set_xlabel('$N$')
axes[0].set_ylabel('Errors of means')
axes[0].set_yscale('log')
axes[0].legend(loc='lower left', fontsize=19)

axes[1].plot(np.arange(2, 16), np.mean(vals_vars, axis=0), color='black', linewidth=2)
axes[1].fill_between(np.arange(2, 16),
                     np.mean(vals_vars, axis=0) - 2 * np.std(vals_vars, axis=0),
                     np.mean(vals_vars, axis=0) + 2 * np.std(vals_vars, axis=0),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[1].plot([2, 15], np.tile(np.mean(errs_variances_pf), (2, )), color='black', linewidth=2, linestyle=':')
axes[1].fill_between([2, 15],
                     np.tile(np.mean(errs_variances_pf), (2, )) - 2 * np.tile(np.std(np.mean(errs_variances_pf, axis=1)), (2, )),
                     np.tile(np.mean(errs_variances_pf), (2, )) + 2 * np.tile(np.std(np.mean(errs_variances_pf, axis=1)), (2, )),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)
axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_xticks(np.arange(2, 16))
axes[1].set_xlabel('$N$')
axes[1].set_ylabel('Errors of variances')
axes[1].set_yscale('log')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0.08)
plt.savefig('convergence_in_N.pdf')
plt.show()
