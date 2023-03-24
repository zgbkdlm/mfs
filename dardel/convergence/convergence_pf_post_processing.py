import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Pack the pf results into a single array. ')
parser.add_argument('--nparticles', type=int, default=100000, help='Number of particle filter particles.')
parser.add_argument('--T', type=int, default=100, help='Time length.')
parser.add_argument('--maxmc', type=int, default=10000, help='Number of Monte Carlo samples.')
args = parser.parse_args()

nparticles, T, num_mc = args.nparticles, args.T, args.maxmc

errs_means = np.zeros((num_mc, T))
errs_variances = np.zeros((num_mc, T))
errs_kls = np.zeros((num_mc, T))

# Load results
for k in range(num_mc):
    filename = f'./results/convergence/pf_{nparticles}_mc_{k}.npz'
    data = np.load(filename)

    errs_means[k] = data['errs_means']
    errs_variances[k] = data['errs_variances']
    errs_kls[k] = data['errs_kls']

# Dump results
filename = f'./results/convergence/pf_{nparticles}.npz'
np.savez(filename, errs_means=errs_means, errs_variances=errs_variances, errs_kls=errs_kls)
