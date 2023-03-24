"""
Load all the PF experiment results and pack them into one .npz.
"""
import argparse
import jax
import numpy as np
from mfs.one_dim.moments import characteristic_from_pdf
from mfs.one_dim.ss_models import benes_bernoulli
from jax.config import config

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Post-processing the Benes--Bernoulli PF results.')
parser.add_argument('--b', type=int, default=2, help='The bounds of the grids for the characteristic function.')
parser.add_argument('--m', type=int, default=2000, help='The number of grids for the characteristic function.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

b, m, num_mcs = args.b, args.m, args.maxmc

dt, T, ts, *_ = benes_bernoulli(2)

# Grids for the characteristic function
zs = np.linspace(-b, b, m)

# Containers for storing the results
pf_filtering_means = np.empty((num_mcs, T))
pf_filtering_cfs = np.empty((num_mcs, T, m), dtype='complex128')

# Load all the results and compute the filtering means and characteristic functions
for k in range(num_mcs):
    print(f'Processing {k} / {num_mcs}')

    # Particle filter solution
    filename = f'./results/benes_bernoulli_pf/b_{b}_m_{m}_mc_{k}.npz'
    data = np.load(filename)

    pf_filtering_means[k] = data['pf_filtering_means']
    pf_filtering_cfs[k] = data['pf_filtering_cfs']

# Dump results
np.savez_compressed(f'./results/benes_bernoulli_pf/b_{b}_m_{m}.npz',
                    pf_filtering_means=pf_filtering_means, pf_filtering_cfs=pf_filtering_cfs)
