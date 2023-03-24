import argparse
import jax
import numpy as np
from mfs.one_dim.moments import characteristic_fn
from mfs.one_dim.ss_models import benes_bernoulli
from jax.config import config

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Post-processing the Benes--Bernoulli experiment results.')
parser.add_argument('--N', type=int, help='Order.')
parser.add_argument('--mode', type=str, default='central', help='Moment filter mode.')
parser.add_argument('--normal', action='store_true', help='Compute the transition moments by Normal approximation. '
                                                          'This is helpful for stability because the approximated '
                                                          'moments are valid by definition.')
parser.add_argument('--b', type=int, default=2, help='The bounds of the grids for the characteristic function.')
parser.add_argument('--m', type=int, default=2000, help='The number of grids for the characteristic function.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

N, mode, b, m, num_mcs = args.N, args.mode, args.b, args.m, args.maxmc

dt, T, ts, *_ = benes_bernoulli(2)
if args.normal:
    _t = '_normal'
else:
    _t = ''

# Grids for the characteristic function
zs = np.linspace(-b, b, m)

# Containers for storing the results
mf_filtering_means = np.empty((num_mcs, T))
mf_filtering_cfs = np.empty((num_mcs, T, m), dtype='complex128')

# Load the results of the moment filter solution
cf_rms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None]), in_axes=[None, 0]))
cf_cms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None, None]), in_axes=[None, 0, 0]))
cf_scms = jax.jit(jax.vmap(jax.vmap(characteristic_fn, in_axes=[0, None, None, None]), in_axes=[None, 0, 0, 0]))

for k in range(num_mcs):
    print(f'{k} / {num_mcs}')

    filename = f'./results/benes_bernoulli_mf/{mode}{_t}_N_{N}_mc_{k}.npz'
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

# Dump results
np.savez_compressed(f'./results/benes_bernoulli_mf/b_{b}_m_{m}_{mode}{_t}_{N}',
                    mf_filtering_means=mf_filtering_means, mf_filtering_cfs=mf_filtering_cfs)
