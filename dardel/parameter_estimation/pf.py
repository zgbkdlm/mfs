"""
Parameter estimation.
"""
import argparse
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
from mfs.classical_filters_smoothers.smc import bootstrap_filter
from mfs.classical_filters_smoothers.resampling import stratified
from mfs.one_dim.ss_models import well_poisson
from jax.config import config

config.update("jax_enable_x64", True)

# Parse CLI arguments
parser = argparse.ArgumentParser(description='The 1D Double-well-Poisson particle filtering parameter estimation.')
parser.add_argument('--nparticles', type=int, default=10000, help='Number of particle filter particles.')
parser.add_argument('--euler', action='store_true', help='Whether use Euler--Maruyama to approximate the transition. '
                                                         'If use TME, the allocated memory is giga big.')
parser.add_argument('--cr', action='store_true', help='Use continuous resampling.')
parser.add_argument('--p1', type=float, default=3., help='The true parameter p1.')
parser.add_argument('--p1_init', type=float, default=0.1, help='The initial guess for the parameter p1.')
parser.add_argument('--p2', type=float, default=3., help='The true parameter p2.')
parser.add_argument('--p2_init', type=float, default=0.1, help='The initial guess for the parameter p2.')
parser.add_argument('--k', type=int, default=0, help='Which Monte Carlo run.')
parser.add_argument('--maxmc', type=int, default=1000, help='The maximum number of Monte Carlo runs.')
args = parser.parse_args()

nparticles, k, num_mcs = args.nparticles, args.k, args.maxmc
true_p1, true_p2 = args.p1, args.p2

# Simulation setting
dt, T, ts, init_cond, drift, dispersion, emission, measurement_cond_pmf, simulate_trajectory = well_poisson(true_p1, 2)


# The objective function
@jax.jit
def obj_func(params, _ys, _key):
    params = jnp.log(jnp.exp(params) + 1.)

    def _drift(x):
        return drift(x, params[0])

    def _measurement_cond_pmf(y, x):
        return measurement_cond_pmf(y, x, params[1])

    if args.euler:
        def state_cond_m_cov(x, _dt):
            return x + _drift(x) * _dt, dispersion(x) ** 2 * _dt
    else:
        def state_cond_m_cov(x, _dt):
            return tme.mean_and_cov(jnp.atleast_1d(x), _dt, _drift, dispersion, order=2)

    def proposal_sampler(x, _key):
        ms, covs = jax.vmap(state_cond_m_cov, in_axes=[0, None])(x, dt)
        return jnp.squeeze(ms) + jnp.squeeze(jnp.sqrt(covs)) * jax.random.normal(_key, (nparticles,))

    if args.cr:
        nell = bootstrap_filter(proposal_sampler, _measurement_cond_pmf, _ys, init_cond.sampler, _key,
                                nparticles, None, conti_resampling=True)[1]
    else:
        nell = bootstrap_filter(proposal_sampler, _measurement_cond_pmf, _ys, init_cond.sampler, _key,
                                nparticles, stratified)[1]
    return nell


# Monte Carlo runs
print(f'Particle filter parameter estimation Monte Carlo run {k} / {num_mcs - 1}')

# Load the random key
key = jnp.asarray(np.load('rng_keys.npy')[k])
key_x0, key_xs, key_ys = jax.random.split(key, 3)
key_pf, _ = jax.random.split(key_ys)

# Simulate a trajectory and measurements
x0 = init_cond.sampler(key_x0, 1)[0]
xs = simulate_trajectory(x0, key_xs)
ys = jax.random.poisson(key_ys, emission(xs, true_p2), (T,))

# Run optimisation
init_params = jnp.log(jnp.exp(jnp.array([args.p1_init, args.p2_init])) - 1.)
opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
opt_params, opt_state = opt_solver.run(init_params, ys, key_pf)
opt_params = jnp.log(jnp.exp(opt_params) + 1.)

# Dump results
filename = f'./results/parameter_estimation_pf{"_cr" if args.cr else ""}/{"euler_" if args.euler else ""}mc_{k}.npz'
np.savez_compressed(filename, success=opt_state.success, opt_params=opt_params)
