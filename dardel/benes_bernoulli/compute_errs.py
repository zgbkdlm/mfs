"""
After all the experiment results are done, run this file to compute the errors.
"""
import argparse
import jax
import numpy as np
from mfs.one_dim.moments import characteristic_fn
from mfs.one_dim.ss_models import benes_bernoulli
from jax.config import config

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Post-processing the Benes--Bernoulli brute force results.')
parser.add_argument('--b', type=int, default=2, help='The bounds of the grids for the characteristic function.')
parser.add_argument('--m', type=int, default=2000, help='The number of grids for the characteristic function.')
parser.add_argument('--N', type=int, help='Order. 2 N - 1 is the highest moment order.')
parser.add_argument('--mode', type=str, default='central', help='Mode of the moment filter. '
                                                                'Options are "raw", "central" (default), and "scaled".')
parser.add_argument('--normal', action='store_true', help='Compute the transition moments by Normal approximation. '
                                                          'This is helpful for stability because the approximated '
                                                          'moments are valid by definition.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

# Parameters and experiment settings
b, m, N, mode, num_mcs = args.b, args.m, args.N, args.mode, args.maxmc
dt, T, ts, *_ = benes_bernoulli(2)
if args.normal:
    _t = '_normal'
else:
    _t = ''
gh_order = 11
tme_order = 3

# Grids for the characteristic function
zs = np.linspace(-b, b, m)

# Load the brute-force solutions
filename = f'../dardel/results/benes_bernoulli_brute_force/b_{b}_m_{m}.npz'
data = np.load(filename)

true_trajs = data['true_trajs']
true_filtering_means = data['true_filtering_means']
true_filtering_cfs = data['true_filtering_cfs']

# Load the Gauss--Hermite solutions
filename = f'../dardel/results/benes_bernoulli_ghf/gh_{gh_order}_tme_{tme_order}.npz'
data = np.load(filename)

ghf_mfs = data['ghf_mfs']
ghf_vfs = data['ghf_vfs']

ghf_filtering_means = ghf_mfs
ghf_filtering_cfs = np.exp(1.j * zs[None, None, :] * ghf_mfs[:, :, None]
                           - 0.5 * zs[None, None, :] ** 2 * ghf_vfs[:, :, None])

# Load the particle filtering solutions
filename = f'../dardel/results/benes_bernoulli_pf/b_{b}_m_{m}.npz'
data = np.load(filename)

pf_filtering_means = data['pf_filtering_means']
pf_filtering_cfs = data['pf_filtering_cfs']

# Load the results of the moment filter solution
mf_filtering_means = np.empty((num_mcs, T))
mf_filtering_cfs = np.empty((num_mcs, T, m), dtype='complex128')

cf_rms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None]), in_axes=[None, 0]))
cf_cms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None, None]), in_axes=[None, 0, 0]))
cf_scms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None, None, None]), in_axes=[None, 0, 0, 0]))

for k in range(num_mcs):
    filename = f'../dardel/results/benes_bernoulli_mf/{mode}{_t}_N_{N}_mc_{k}.npz'
    data = np.load(filename)

    if mode == 'raw':
        rmss = data['rmss']
        mf_filtering_means[k] = rmss[:, 1]
        mf_filtering_cfs[k] = cf_rms(zs, rmss)

    elif mode == 'central':
        cmss, means = data['cmss'], data['means']
        mf_filtering_means[k] = means
        mf_filtering_cfs[k] = cf_cms(zs, cmss, means)

    elif mode == 'scaled':
        scmss, means, scales = data['scmss'], data['means'], data['scales']
        mf_filtering_means[k] = means
        mf_filtering_cfs[k] = cf_scms(zs, scmss, means, scales)
    else:
        raise ValueError(f'Mode {mode} not recognised.')

# Compute errors
true_trajs_vs_true_means_abs = np.abs(true_trajs - true_filtering_means)
true_trajs_vs_mf_means_abs = np.abs(true_trajs - mf_filtering_means)
true_trajs_vs_ghf_means_abs = np.abs(true_trajs - ghf_filtering_means)
true_trajs_vs_pf_means_abs = np.abs(true_trajs - pf_filtering_means)

true_means_vs_mf_means_abs = np.abs(true_filtering_means - mf_filtering_means)
true_means_vs_ghf_means_abs = np.abs(true_filtering_means - ghf_filtering_means)
true_means_vs_pf_means_abs = np.abs(true_filtering_means - pf_filtering_means)

true_cfs_vs_mf_l1 = np.trapz(np.abs(true_filtering_cfs - mf_filtering_cfs), zs, axis=-1)
true_cfs_vs_mf_l2 = np.sqrt(np.trapz(np.abs(true_filtering_cfs - mf_filtering_cfs) ** 2, zs, axis=-1))
true_cfs_vs_mf_sup = np.max(np.abs(true_filtering_cfs - mf_filtering_cfs), axis=-1)

true_cfs_vs_ghf_l1 = np.trapz(np.abs(true_filtering_cfs - ghf_filtering_cfs), zs, axis=-1)
true_cfs_vs_ghf_l2 = np.sqrt(np.trapz(np.abs(true_filtering_cfs - ghf_filtering_cfs) ** 2, zs, axis=-1))
true_cfs_vs_ghf_sup = np.max(np.abs(true_filtering_cfs - ghf_filtering_cfs), axis=-1)

true_cfs_vs_pf_l1 = np.trapz(np.abs(true_filtering_cfs - pf_filtering_cfs), zs, axis=-1)
true_cfs_vs_pf_l2 = np.sqrt(np.trapz(np.abs(true_filtering_cfs - pf_filtering_cfs) ** 2, zs, axis=-1))
true_cfs_vs_pf_sup = np.max(np.abs(true_filtering_cfs - pf_filtering_cfs), axis=-1)

np.savez_compressed(f'./results/benes_bernoulli_errs/errs_b_{b}_m_{m}_{mode}{_t}_N_{N}.npz',
                    true_trajs_vs_true_means_abs=true_trajs_vs_true_means_abs,
                    true_trajs_vs_mf_means_abs=true_trajs_vs_mf_means_abs,
                    true_trajs_vs_ghf_means_abs=true_trajs_vs_ghf_means_abs,
                    true_trajs_vs_pf_means_abs=true_trajs_vs_pf_means_abs,
                    true_means_vs_mf_means_abs=true_means_vs_mf_means_abs,
                    true_means_vs_ghf_means_abs=true_means_vs_ghf_means_abs,
                    true_means_vs_pf_means_abs=true_means_vs_pf_means_abs,
                    true_cfs_vs_mf_l1=true_cfs_vs_mf_l1, true_cfs_vs_mf_l2=true_cfs_vs_mf_l2,
                    true_cfs_vs_mf_sup=true_cfs_vs_mf_sup,
                    true_cfs_vs_ghf_l1=true_cfs_vs_ghf_l1, true_cfs_vs_ghf_l2=true_cfs_vs_ghf_l2,
                    true_cfs_vs_ghf_sup=true_cfs_vs_ghf_sup,
                    true_cfs_vs_pf_l1=true_cfs_vs_pf_l1, true_cfs_vs_pf_l2=true_cfs_vs_pf_l2,
                    true_cfs_vs_pf_sup=true_cfs_vs_pf_sup)
