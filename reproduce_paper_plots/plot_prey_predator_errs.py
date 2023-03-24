import numpy as np
import matplotlib.pyplot as plt
from mfs.multi_dims.ss_models import prey_predator

dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(((0, 0), (0, 1)))

# Settings
transition_mf = 'tme_2'
transition_others = 'tme_2'
Ns = [5, 7]
markers = ['x', 'v']
mode = 'central'
maxmc = 10000

# # Load EKF GHF
# errs_ekf = np.zeros((maxmc, T, 2))
# errs_ghf = np.zeros((maxmc, T, 2))
# for k in range(maxmc):
#     filename = f'../dardel/results/prey_predator_ghf_ekf/{transition_others}_mc_{k}.npz'
#     data = np.load(filename)
#
#     errs_ekf[k] = data['errs_ekf']
#     errs_ghf[k] = data['errs_ghf']
#
# num_nans = np.any(np.isnan(errs_ekf), axis=(1, 2))
# errs_ekf = errs_ekf[~num_nans]
# print(f'EKF divergences: {np.count_nonzero(num_nans)}')
# print(f'EKF filter errs: {np.mean(np.sum(errs_ekf, axis=-1))}, {np.std(np.sum(errs_ekf, axis=-1))}')
#
# num_nans = np.any(np.isnan(errs_ghf), axis=(1, 2))
# errs_ghf = errs_ghf[~num_nans]
# print(f'GHF divergences: {np.count_nonzero(num_nans)}')
# print(f'GHF filter errs: {np.mean(np.sum(errs_ghf, axis=-1))}, {np.std(np.sum(errs_ghf, axis=-1))}')

# Load particle filter
errs_pf = np.zeros((maxmc, T, 2))
for k in range(maxmc):
    filename = f'../dardel/results/prey_predator_pf/{transition_others}_mc_{k}.npz'
    data = np.load(filename)

    errs_pf[k] = data['errs']

num_nans = np.any(np.isnan(errs_pf), axis=(1, 2))
errs_pf = errs_pf[~num_nans]
print(f'PF divergences: {np.count_nonzero(num_nans)}')
print(f'PF filter errs: {np.mean(np.sum(errs_pf, axis=-1))}, {np.std(np.sum(errs_pf, axis=-1))}')

# Load Cramer--Rao
# crlbs = np.load('../dardel/results/prey_predator_cramer_rao/crlbs.npz')['crlbs']

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

zoom = 10
plt.figure(figsize=(7, 4))

plt.plot(ts[::zoom], np.mean(np.sum(errs_pf, axis=-1), axis=0)[::zoom], c='black', linestyle='--',
         label='Particle filter', alpha=0.8)

for N, marker in zip(Ns, markers):
    # Load moment filter
    errs_mf = np.zeros((maxmc, T, 2))
    for k in range(maxmc):
        filename = f'../dardel/results/prey_predator_mf/{mode}_{transition_mf}_N_{N}_mc_{k}.npz'
        data = np.load(filename)
        errs_mf[k] = data['errs']

    num_nans = np.any(np.isnan(errs_mf), axis=(1, 2))
    errs_mf = errs_mf[~num_nans]
    print(f'{N}-order Moment filter divergences: {np.count_nonzero(num_nans)}')
    print(f'{N}-order Moment filter errs: {np.mean(np.sum(errs_mf, axis=-1))}, {np.std(np.sum(errs_mf, axis=-1))}')

    plt.plot(ts[::zoom], np.mean(np.sum(errs_mf, axis=-1), axis=0)[::zoom], c='black',
             marker=marker, markevery=20, markersize=10, markerfacecolor='none', markeredgecolor='black',
             label=f'Moment filter ($N={N}$)')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel(r'Absolute error')

plt.legend(fontsize=18)
plt.tight_layout(pad=0.1)
plt.savefig('prey_predator_errs.pdf')
plt.show()
