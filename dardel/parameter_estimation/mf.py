"""
Parameter estimation.
"""
import argparse
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
from mfs.one_dim.filtering import moment_filter_cms
from mfs.one_dim.moments import sde_cond_moments_euler, sde_cond_moments_tme_normal
from mfs.one_dim.ss_models import well_poisson
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Double-well-Poisson moment filtering parameter estimation.')
parser.add_argument('--N', type=int, help='Order. 2 N - 1 is the highest moment order.')
parser.add_argument('--euler', action='store_true', help='Whether use Euler--Maruyama to approximate the transition '
                                                         'moments. Using TME here can be super slow, I dont know why.')
parser.add_argument('--p1', type=float, default=3., help='The true parameter p1.')
parser.add_argument('--p1_init', type=float, default=0.1, help='The initial guess for the parameter p1.')
parser.add_argument('--p2', type=float, default=3., help='The true parameter p2.')
parser.add_argument('--p2_init', type=float, default=0.1, help='The initial guess for the parameter p2.')
parser.add_argument('--k', type=int, default=0, help='Which Monte Carlo run.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

N, k, num_mcs = args.N, args.k, args.maxmc
true_p1, true_p2 = args.p1, args.p2

# Simulation setting
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate_trajectory = well_poisson(true_p1, N)


# The objective function
@jax.jit
def obj_func(params, _ys):
    params = jnp.log(jnp.exp(params) + 1.)

    def _drift(x):
        return drift(x, params[0])

    def _measurement_cond_pmf(y, x):
        return measurement_cond_pmf(y, x, params[1])

    if args.euler:
        _, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_euler(_drift, dispersion, dt, N)
    else:
        _, sde_cond_cms, _, state_cond_mean, _ = sde_cond_moments_tme_normal(_drift, dispersion, dt, 2, N)

    _, _, nell = moment_filter_cms(sde_cond_cms, state_cond_mean, _measurement_cond_pmf,
                                   init_cond.cms, init_cond.mean, _ys)
    return nell


# Monte Carlo runs
print(f'{N}-order moment filter parameter estimation Monte Carlo run {k} / {num_mcs - 1}')

# Load the random key
key = jnp.asarray(np.load('rng_keys.npy')[k])
key_x0, key_xs, key_ys = jax.random.split(key, 3)

# Simulate a trajectory and measurements
x0 = init_cond.sampler(key_x0, 1)[0]
xs = simulate_trajectory(x0, key_xs)
ys = jax.random.poisson(key_ys, emission(xs, true_p2), (T,))

# Run optimisation
init_params = jnp.log(jnp.exp(jnp.array([args.p1_init, args.p2_init])) - 1.)
opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
opt_params, opt_state = opt_solver.run(init_params, ys)
opt_params = jnp.log(jnp.exp(opt_params) + 1.)

# Dump results
filename = f'./results/parameter_estimation_mf/N_{N}{"_euler" if args.euler else ""}_mc_{k}.npz'
np.savez_compressed(filename, success=opt_state.success, opt_params=opt_params)
