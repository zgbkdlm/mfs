import numpy as np
import pytest
from mfs.multi_dims.multi_indices import generate_graded_lexico_multi_indices, \
    graded_lexico_indexof_multi_index, sizeof_multi_indices

np.random.seed(666)


@pytest.mark.parametrize('dim', [1, 3, 5])
@pytest.mark.parametrize('upper_sum', [4, 6])
@pytest.mark.parametrize('lower_sum', [0, 3])
class TestMultiIndices:

    def test_generate_multi_indices(self, dim, upper_sum, lower_sum):
        """Verify by the definition of graded lexicographic order.
        """
        multi_indices = generate_graded_lexico_multi_indices(d=dim,
                                                             upper_sum=upper_sum,
                                                             lower_sum=lower_sum)
        for i in range(1, multi_indices.shape[0]):
            x = multi_indices[i - 1]
            y = multi_indices[i]

            assert sum(y) >= sum(x)

            if sum(y) == sum(x):
                diff = y - x
                assert diff[np.min(np.nonzero(diff))] > 0

    def test_indexing(self, dim, upper_sum, lower_sum):
        """Test if the indexing method gives the correct index
        """
        multi_indices = generate_graded_lexico_multi_indices(d=dim,
                                                             upper_sum=upper_sum,
                                                             lower_sum=lower_sum)
        ind = np.random.choice(np.arange(multi_indices.shape[0]))
        multi_index = multi_indices[ind]

        assert graded_lexico_indexof_multi_index(multi_index, lower_sum=lower_sum) == ind

    def test_sizing(self, dim, upper_sum, lower_sum):
        """Test the sizing method.
        """
        multi_indices = generate_graded_lexico_multi_indices(d=dim,
                                                             upper_sum=upper_sum,
                                                             lower_sum=lower_sum)

        assert sizeof_multi_indices(dim, upper_sum, lower_sum) == multi_indices.shape[0]
