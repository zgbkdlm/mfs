"""
Load all the brute force experiment results and pack them into one .npz.
"""
import argparse
import jax
import numpy as np
from mfs.one_dim.moments import characteristic_from_pdf
from mfs.one_dim.ss_models import benes_bernoulli
from jax.config import config

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Post-processing the Benes--Bernoulli brute force results.')
parser.add_argument('--b', type=int, default=2, help='The bounds of the grids for the characteristic function.')
parser.add_argument('--m', type=int, default=2000, help='The number of grids for the characteristic function.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

b, m, num_mcs = args.b, args.m, args.maxmc

dt, T, ts, *_ = benes_bernoulli(2)

# Grids for the characteristic function
zs = np.linspace(-b, b, m)


@jax.jit
def cfs_fn(_true_pdfs, _grids):
    return jax.vmap(jax.vmap(characteristic_from_pdf, in_axes=[0, None, None]),
                    in_axes=[None, 0, None])(zs, _true_pdfs, _grids)


# Containers for storing the results
true_trajs = np.empty((num_mcs, T))
true_filtering_means = np.empty((num_mcs, T))
true_filtering_cfs = np.empty((num_mcs, T, m), dtype='complex128')

# Load all the results and compute the filtering means and characteristic functions
for k in range(num_mcs):
    print(f'Processing {k} / {num_mcs}')

    # Brute force solution
    filename = f'./results/benes_bernoulli_brute_force/mc_{k}.npz'
    data = np.load(filename)
    spatial_grids = data['spatial_grids']
    true_pdfs = data['true_pdfs']

    true_trajs[k] = data['xs']
    true_filtering_means[k] = np.trapz(spatial_grids[None, :] * true_pdfs, spatial_grids, axis=1)
    true_filtering_cfs[k] = cfs_fn(true_pdfs, spatial_grids)

# Dump results
np.savez_compressed(f'./results/benes_bernoulli_brute_force/b_{b}_m_{m}.npz',
                    true_trajs=true_trajs, true_filtering_means=true_filtering_means,
                    true_filtering_cfs=true_filtering_cfs)
