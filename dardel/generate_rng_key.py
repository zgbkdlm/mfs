"""
Generate a bunch of random keys to get independent Monte Carlo seeds.
"""
import jax
import numpy as np

max_mcs = 1000000

key = jax.random.PRNGKey(999)
keys = jax.random.split(key, max_mcs)

np.save('./rng_keys.npy', np.asarray(keys))
