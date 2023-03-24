# Copyright (C) 2022 Zheng Zhao
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Manipulations for multi-indices.
"""
import math
import numpy as np
from typing import Sequence
from functools import partial


def sizeof_multi_indices(d: int, upper_sum: int, lower_sum: int = 0) -> int:
    r"""Cardinality/counts/size of collection of a d-dimensional multi-indices

    .. math::

        \{ x: lower sum \leq |x| \leq upper sum \},

    where the multi-indices are of the form :math:`x = (x_1, x_2, \ldots, x_d)`. If the lower sum is greater than
    the upper sum, the function returns zero.

    Parameters
    ----------
    d : int
        Dimension of the multi-indices. Must be >= 1.
    upper_sum : int
        Upper sum. 
    lower_sum : int, default=0
        Lower sum. Default is zero.

    Returns
    -------
    int
        The cardinality/counts/size of :math:`{x: lower sum \leq |x| \leq upper sum}`.
    """
    if upper_sum == lower_sum:
        return math.comb(upper_sum + d - 1, upper_sum)

    if upper_sum < lower_sum:
        return 0

    if lower_sum == 0:
        return math.comb(upper_sum + d, upper_sum)
    else:
        return math.comb(upper_sum + d, upper_sum) - math.comb(lower_sum - 1 + d, lower_sum - 1)


def graded_lexico_indexof_multi_index(multi_index: Sequence[int], lower_sum: int = 0) -> int:
    r"""Given a collection of multi-indices

    .. math::

        \{ x: lower sum \leq |x| \leq upper sum \},

    that are graded lexicographic ordered. Suppose that :math:`x` is a multi-index in the collection. This function
    finds the index of :math:`x` in this collection.

    Please note that the index is Pythonic, that is, it starts from 0 not 1.

    Parameters
    ----------
    multi_index : Sequence (d, )
        Any sequence of `d` integers that has length and supports sum, for example, `list` and `np.ndarray`.
    lower_sum : int, default=0
        Lower sum. Must be non-negative, and the default is zero.

    Returns
    -------
    int
        The Python index of the `multi_index` in the collection.

    Notes
    -----
    Definition of graded lexicographic order. Let `x` and `y` be two d-dimensional multi-indices, we say that `x > y`
    in the graded lexicographic order sense, if `|x| > |y|` , or the first non-zero entry (begins from left) of the
    difference multi-index `x - y` is positive and `|x| = |y|`. See the monograph by Charles F. Dunkl and Yuan Xu,
    2014, pp. 59 for details.
    """
    d = len(multi_index)

    total_sum = sum(multi_index)
    base_pos = sizeof_multi_indices(d, total_sum - 1, 0)

    pos = base_pos
    sub_sum = total_sum

    for i in range(d):
        ith_index = multi_index[i]

        if ith_index >= 1:
            l, u = sub_sum - ith_index + 1, sub_sum
            pos += sizeof_multi_indices(d - (i + 1), u, l)

        sub_sum -= ith_index

    if lower_sum != 0:
        return pos - sizeof_multi_indices(d, lower_sum - 1)
    else:
        return pos


def _next_graded_lexico_multi_index(multi_index: np.ndarray) -> np.ndarray:
    """A silly but simple way to iterate multi-index. Don't use if you care about performance.
    """
    d = len(multi_index)
    total_sum = sum(multi_index)

    if not np.any(multi_index[1:]):
        next_multi_index = np.zeros_like(multi_index)
        next_multi_index[-1] = total_sum + 1
    else:
        next_multi_index = multi_index.copy()
        last_nonzero_ind = np.max(np.nonzero(multi_index))

        if last_nonzero_ind == d - 1:
            next_multi_index[-1] -= 1
            next_multi_index[-2] += 1
        else:
            next_multi_index[last_nonzero_ind] = 0
            next_multi_index[last_nonzero_ind - 1] += 1
            next_multi_index[-1] += multi_index[last_nonzero_ind] - 1

    return next_multi_index


def generate_graded_lexico_multi_indices(d: int, upper_sum: int, lower_sum: int = 0) -> np.ndarray:
    r"""Generate a set of d-dimensional multi-indices

    .. math::

        \{ x: lower sum \leq |x| \leq upper sum \},

    ordered in the graded lexicographic sense. Return the set as a numpy array of shape `(z, d)`, where `z` is the
    total number of multi-indices.

    Parameters
    ----------
    d : int
        Dimension of the multi-indices. Must >= 1.
    upper_sum : int
        The upper sum of the collection. Must >= `lower_sum`.
    lower_sum : int, default=0
        The lower sum of the collection. The default is zero.

    Returns
    -------
    np.ndarray (z, d)
        A 2D numpy array (int64) of the collection.
    """
    if d == 1:
        return np.arange(lower_sum, upper_sum + 1).reshape((upper_sum - lower_sum + 1, 1))

    z = sizeof_multi_indices(d, upper_sum, lower_sum)
    multi_indices = np.zeros((z, d), dtype='int64')

    init_multi_index = np.zeros((d,))
    init_multi_index[-1] = lower_sum

    multi_indices[0, :] = init_multi_index

    for i in range(1, z):
        multi_indices[i, :] = _next_graded_lexico_multi_index(multi_indices[i - 1, :])

    return multi_indices


@partial(np.vectorize, signature='(d)->()')
def find_indices(multi_index):
    return graded_lexico_indexof_multi_index(multi_index, lower_sum=0)


def gram_and_hankel_indices_graded_lexico(N: int, d: int) -> np.ndarray:
    r"""Let

    .. math::

        M^N = \{ m^n : |n| \leq 2 \, N - 1 \}

    be a collection of moments powered by d-dimensional multi-indices, and the moments are ordered in the graded
    lexicographic sense. This function generates d + 1 matrices of indices, so that we can generate the Gram and
    multiplication matrices `G, H1, H2, ... Hd` by evaluating the moments array with the generated indices. For
    example, if `ms` is the numpy array of the collection of the moments, and `inds` is the return of this function.
    Then `G = ms[inds[0]]` is the Gram matrix, and `Hs = ms[inds[1:]]` are the d Hankel matrices.

    Parameters
    ----------
    N : int
        N is such that 2 N - 1 is the highest order of moment.
    d : int
        Dimension of the moment.

    Returns
    -------
    np.ndarray (d + 1, s, s)
        A numpy arrays of indices, where `s = comb(N - 1 + d, N - 1)`. The first is for the Gram matrix, where the rest
        are for the d Hankel matrices.

    Notes
    -----
    Do not jit this function. The function outputs ought to be the compilation constants to jax.
    """
    s = math.comb(N - 1 + d, N - 1)
    inds = np.zeros((d + 1, s, s), dtype='int64')

    # Gram matrix indices
    basis_multi_indices = generate_graded_lexico_multi_indices(d, upper_sum=N - 1, lower_sum=0)
    gram_multi_indices = basis_multi_indices[:, None, :] + basis_multi_indices[None, :, :]
    inds[0] = find_indices(gram_multi_indices)

    # Multiplication matrices indices
    for i in range(d):
        gram_multi_indices[:, :, i] += 1
        inds[i + 1] = find_indices(gram_multi_indices)
        gram_multi_indices[:, :, i] -= 1

    return inds
