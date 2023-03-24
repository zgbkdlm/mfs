"""
Prey-predator experiments.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, \
    gram_and_hankel_indices_graded_lexico
from mfs.multi_dims.moments import sde_cond_moments_euler_maruyama, sde_cond_moments_tme_normal, sde_cond_moments_tme
from mfs.multi_dims.filtering import moment_filter_nd_rms, moment_filter_nd_cms
from mfs.multi_dims.ss_models import prey_predator
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 2D prey-predator moment filtering.')
parser.add_argument('--N', type=int, help='Order.')
parser.add_argument('--trans', type=str, default='euler', help='The method to compute the transition moments.  '
                                                               'Options are "euler", "tme_normal_*", and "tme_*".')
parser.add_argument('--mode', type=str, default='central', help='Mode of the moment filter. '
                                                                'Options are "raw", "central" (default), and "scaled".')
parser.add_argument('--st_mc', type=int, default=0, help='The start MC run.')
parser.add_argument('--ed_mc', type=int, help='The end MC run.')
args = parser.parse_args()

d = 2
N, transition, mode = args.N, args.trans, args.mode

# Multi-indices
multi_indices = generate_graded_lexico_multi_indices(d, 2 * N - 1, 0)
inds = gram_and_hankel_indices_graded_lexico(N, d)

# Model
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate = prey_predator(multi_indices)
rms0, cms0, mean0 = init_cond.rms, init_cond.cms, init_cond.mean

if 'euler' in transition:
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_euler_maruyama(drift, dispersion,
                                                                                        dt, multi_indices)
    indexing = 'index'
elif 'tme_normal' in transition:
    tme_order = int(transition.split('_')[-1])
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_tme_normal(drift, dispersion,
                                                                                    dt, tme_order, multi_indices)
    indexing = 'index'
else:
    tme_order = int(transition.split('_')[-1])
    sde_cond_rms, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_tme(drift, dispersion, dt, tme_order)
    indexing = 'multi-index'

# Create filter
if mode == 'raw':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_nd_rms((sde_cond_rms, indexing), measurement_cond_pmf, _ys,
                                    (jnp.asarray(multi_indices), inds), rms0)
elif mode == 'central':
    @jax.jit
    def moment_filter(_ys):
        return moment_filter_nd_cms((sde_cond_cms, indexing), state_cond_mean, measurement_cond_pmf, _ys,
                                    (jnp.asarray(multi_indices), inds), cms0, mean0)
else:
    raise NotImplementedError(f'Mode {mode} not implemented.')

# Monte Carlo
for k in range(args.st_mc, args.ed_mc + 1):
    print(f'{N}-order ({mode}) moment filter Monte Carlo run {k} / ({args.st_mc} to {args.ed_mc})')

    # Load the random key
    key = jnp.asarray(np.load('rng_keys.npy')[k])

    # Simulate a trajectory and measurements
    _, xs, ys = simulate(key)

    while np.any(np.isnan(xs)):
        key, _ = jax.random.split(key)
        _, xs, ys = simulate(key)

    # Moment filtering
    filename = f'./results/prey_predator_mf/{mode}_{transition}_N_{N}_mc_{k}.npz'

    if mode == 'raw':
        rmss, _ = moment_filter(ys)
        means = np.concatenate([rmss[:, 2, None], rmss[:, 1, None]], axis=-1)
    elif mode == 'central':
        _, means, _ = moment_filter(ys)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented.')

    np.savez_compressed(filename, errs=np.abs(xs - means))
