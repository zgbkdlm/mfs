import numpy as np
import matplotlib.pyplot as plt

# Settings
num_mcs = 1000
N = 7
bounds = [0., 10.]


def find_valid(stats, vals):
    # Results that converge
    num_divergences = np.count_nonzero(~stats)
    vals = vals[stats, :]

    # Results that not hit the optimisation bounds
    masks = np.any(vals >= bounds[1], axis=1)
    num_divergences += np.count_nonzero(masks)
    return num_divergences, vals[~masks, :]


# Load the results from GHF and EKF
opt_stats_ghf = np.empty((num_mcs,), dtype='bool')
opt_params_ghf = np.empty((num_mcs, 2))
opt_stats_ekf = np.empty((num_mcs,), dtype='bool')
opt_params_ekf = np.empty((num_mcs, 2))

for k in range(num_mcs):
    filename = f'../dardel/results/parameter_estimation_ghf_ekf/gh_11_mc_{k}.npz'
    data = np.load(filename)

    opt_stats_ghf[k] = data['success']
    opt_params_ghf[k] = data['opt_params']

    filename = f'../dardel/results/parameter_estimation_ghf_ekf/ekf_mc_{k}.npz'
    data = np.load(filename)

    opt_stats_ekf[k] = data['success']
    opt_params_ekf[k] = data['opt_params']

divergences_ghf, opt_params_ghf = find_valid(opt_stats_ghf, opt_params_ghf)
print(divergences_ghf)

divergences_ekf, opt_params_ekf = find_valid(opt_stats_ekf, opt_params_ekf)
print(divergences_ekf)

# Load the results from the moment filter
opt_stats_mf = np.empty((num_mcs,), dtype='bool')
opt_params_mf = np.empty((num_mcs, 2))

for k in range(num_mcs):
    filename = f'../dardel/results/parameter_estimation_mf/N_{N}_mc_{k}.npz'
    data = np.load(filename)

    opt_stats_mf[k] = data['success']
    opt_params_mf[k] = data['opt_params']

divergences_mf, opt_params_mf = find_valid(opt_stats_mf, opt_params_mf)
print(divergences_mf)

# Load the results from the particle filter
opt_stats_pf = np.empty((num_mcs,), dtype='bool')
opt_params_pf = np.empty((num_mcs, 2))

for k in range(num_mcs):
    filename = f'../dardel/results/parameter_estimation_pf/mc_{k}.npz'
    data = np.load(filename)

    opt_stats_pf[k] = data['success']
    opt_params_pf[k] = data['opt_params']

divergences_pf, opt_params_pf = find_valid(opt_stats_pf, opt_params_pf)
print(divergences_pf)

# Load the results from the particle filter with continuous resampling
opt_stats_pf_cr = np.empty((num_mcs,), dtype='bool')
opt_params_pf_cr = np.empty((num_mcs, 2))

for k in range(num_mcs):
    filename = f'../dardel/results/parameter_estimation_pf_cr/mc_{k}.npz'
    data = np.load(filename)

    opt_stats_pf_cr[k] = data['success']
    opt_params_pf_cr[k] = data['opt_params']

divergences_pf_cr, opt_params_pf_cr = find_valid(opt_stats_pf_cr, opt_params_pf_cr)
print(divergences_pf_cr)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})


def set_violins(_violins):
    for body in _violins['bodies']:
        body.set_facecolor('black')
        body.set_edgecolor('none')
        body.set_alpha(0.2)

    for part in ['cbars', 'cmins', 'cmaxes']:
        _violins[part].set_edgecolor('black')
        _violins[part].set_linestyle('--')
        _violins[part].set_alpha(0.3)

    _violins['cmeans'].set_color('black')
    _violins['cmeans'].set_linestyle(':')
    _violins['cmeans'].set_linewidth(2)

    _violins['cmedians'].set_color('black')
    _violins['cmedians'].set_linewidth(2)


fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

# For theta_1
violins = axes[0].violinplot([opt_params_ghf[:, 0], opt_params_ekf[:, 0], opt_params_mf[:, 0],
                              opt_params_pf[:, 0], opt_params_pf_cr[:, 0]],
                             showmeans=True, showmedians=True)
set_violins(violins)
axes[0].hlines(3, 0.8, 5.2, colors='black', linewidth=1, alpha=0.7)
axes[0].grid(linestyle='--', alpha=0.3, which='both')
axes[0].set_ylabel(r'Estimated $\theta_1$')
axes[0].set_xticks([1, 2, 3, 4, 5], ['GHF', 'EKF', 'MF', 'PF', 'PFC'])

# For theta_2
violins = axes[1].violinplot([opt_params_ghf[:, 1], opt_params_ekf[:, 1], opt_params_mf[:, 1],
                              opt_params_pf[:, 1], opt_params_pf_cr[:, 1]],
                             showmeans=True, showmedians=True)
set_violins(violins)
axes[1].hlines(3, 0.8, 5.2, colors='black', linewidth=1, alpha=0.7)
axes[1].grid(linestyle='--', alpha=0.3, which='both')
axes[1].set_ylabel(r'Estimated $\theta_2$')
axes[1].set_xticks([1, 2, 3, 4, 5], ['GHF', 'EKF', 'MF', 'PF', 'PFC'])

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0.133)
plt.savefig('param_est.pdf')
plt.show()
