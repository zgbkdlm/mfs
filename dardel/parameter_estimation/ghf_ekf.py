"""
Parameter estimation.
"""
import argparse
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.one_dim.ss_models import well_poisson
from mfs.classical_filters_smoothers.gfs import sgp_filter, ekf
from mfs.classical_filters_smoothers.quadratures import SigmaPoints
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Double-well-Poisson GH filtering parameter estimation.')
parser.add_argument('--gh', type=int, default=11, help='The order of the Gauss--Hermite integration.')
parser.add_argument('--euler', action='store_true', help='Whether use Euler--Maruyama to approximate the transition.')
parser.add_argument('--p1', type=float, default=3., help='The true parameter p1.')
parser.add_argument('--p1_init', type=float, default=0.1, help='The initial guess for the parameter p1.')
parser.add_argument('--p2', type=float, default=3., help='The true parameter p2.')
parser.add_argument('--p2_init', type=float, default=0.1, help='The initial guess for the parameter p2.')
parser.add_argument('--k', type=int, default=0, help='Which Monte Carlo run.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

gh_order, k, num_mcs = args.gh, args.k, args.maxmc
true_p1, true_p2 = args.p1, args.p2

# Simulation setting
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate_trajectory = well_poisson(true_p1, 2)

# Sigma points
sgps = SigmaPoints.gauss_hermite(d=1, order=gh_order)


# The objective function
def obj_func(params, _ys, method):
    params = jnp.log(jnp.exp(params) + 1.)

    def _drift(x):
        return drift(x, params[0])

    def measurement_cond_m_cov(x):
        lam = emission(x, params[1])
        return jnp.atleast_1d(lam), jnp.atleast_2d(lam)

    if args.euler:
        def state_cond_m_cov(x, _dt):
            return jnp.atleast_1d(x + _drift(x) * _dt), jnp.atleast_2d(dispersion(x) ** 2 * dt)
    else:
        def state_cond_m_cov(x, _dt):
            return tme.mean_and_cov(jnp.atleast_1d(x), _dt, _drift, dispersion, order=2)

    if method == 'ghf':
        _, _, nells = sgp_filter(state_cond_m_cov, measurement_cond_m_cov, sgps,
                                 jnp.atleast_1d(init_cond.mean), jnp.atleast_2d(init_cond.variance), dt,
                                 _ys[:, None], const_measurement_cov=False)
    else:
        _, _, nells = ekf(state_cond_m_cov, measurement_cond_m_cov,
                          jnp.atleast_1d(init_cond.mean), jnp.atleast_2d(init_cond.variance), dt, _ys[:, None])

    return nells[-1]


@jax.jit
def obj_func_ghf(params, _ys):
    return obj_func(params, _ys, 'ghf')


@jax.jit
def obj_func_ekf(params, _ys):
    return obj_func(params, _ys, 'ekf')


# Monte Carlo runs
print(f'Gaussian filter parameter estimation Monte Carlo run {k} / {num_mcs - 1}')

# Load the random key
key = jnp.asarray(np.load('rng_keys.npy')[k])
key_x0, key_xs, key_ys = jax.random.split(key, 3)

# Simulate a trajectory and measurements
x0 = init_cond.sampler(key_x0, 1)[0]
xs = simulate_trajectory(x0, key_xs)
ys = jax.random.poisson(key_ys, emission(xs, true_p2), (T,))

# Run optimisation of GHF
init_params = jnp.log(jnp.exp(jnp.array([args.p1_init, args.p2_init])) - 1.)
opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func_ekf)
opt_params, opt_state = opt_solver.run(init_params,  ys)
opt_params = jnp.log(jnp.exp(opt_params) + 1.)

# Dump results
filename = f'./results/parameter_estimation_ghf_ekf/gh_{gh_order}{"_euler" if args.euler else ""}_mc_{k}.npz'
np.savez_compressed(filename, success=opt_state.success, opt_params=opt_params)

# Run optimisation of EKF
opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func_ekf)
opt_params, opt_state = opt_solver.run(init_params, ys)
opt_params = jnp.log(jnp.exp(opt_params) + 1.)

# Dump results
filename = f'./results/parameter_estimation_ghf_ekf/ekf{"_euler" if args.euler else ""}_mc_{k}.npz'
np.savez_compressed(filename, success=opt_state.success, opt_params=opt_params)
