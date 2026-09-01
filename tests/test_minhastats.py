import math
import pytest
import numpy as np
from scipy import stats
from src import minhastats


@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=10, size=500).tolist()
    y = [2 * xi + 5 + np.random.normal(0, 2) for xi in x]
    return x, y


def test_stats_functions(sample_data):
    x, _ = sample_data
    tol = 1e-5
    assert math.isclose(minhastats.mean(x), np.mean(x), abs_tol=tol)
    assert math.isclose(minhastats.median(x), np.median(x), abs_tol=tol)
    assert math.isclose(minhastats.variance(x, ddof=0), np.var(x, ddof=0), abs_tol=tol)
    assert math.isclose(minhastats.variance(x, ddof=1), np.var(x, ddof=1), abs_tol=tol)
    assert math.isclose(minhastats.std_dev(x, ddof=0), np.std(x, ddof=0), abs_tol=tol)
    assert math.isclose(minhastats.std_dev(x, ddof=1), np.std(x, ddof=1), abs_tol=tol)
    assert math.isclose(minhastats.quantile(x, 0.25), np.percentile(x, 25), abs_tol=1e-2)
    assert math.isclose(minhastats.percentile(x, 90), np.percentile(x, 90), abs_tol=1e-2)
    assert math.isclose(minhastats.amplitude(x), max(x) - min(x), abs_tol=tol)
    assert math.isclose(minhastats.coefficient_of_variation(x), np.std(x, ddof=1) / abs(np.mean(x)) * 100, abs_tol=tol)


def test_mode_known_values():
    assert minhastats.mode([1, 2, 2, 3, 3, 3, 4]) == [3]
    assert minhastats.mode([1, 1, 2, 2, 3]) == [1, 2]


def test_correlation_and_regression(sample_data):
    x, y = sample_data
    tol = 1e-5
    r_proprio = minhastats.pearson_correlation(x, y)
    r_scipy, _ = stats.pearsonr(x, y)
    assert math.isclose(r_proprio, r_scipy, abs_tol=tol)

    b0, b1, r2 = minhastats.linear_regression(x, y)
    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    assert math.isclose(b1, slope, abs_tol=tol)
    assert math.isclose(b0, intercept, abs_tol=tol)
    assert math.isclose(r2, r_val**2, abs_tol=tol)


def test_covariance_population_and_sample(sample_data):
    x, y = sample_data
    assert math.isclose(minhastats.covariance(x, y, ddof=0), np.cov(x, y, ddof=0)[0, 1], abs_tol=1e-5)
    assert math.isclose(minhastats.covariance(x, y, ddof=1), np.cov(x, y, ddof=1)[0, 1], abs_tol=1e-5)


def test_variance_matches_precomputed_mean(sample_data):
    x, _ = sample_data
    m = minhastats.mean(x)
    assert math.isclose(minhastats.variance(x, ddof=1), minhastats.variance(x, ddof=1, _mean=m), abs_tol=1e-12)


def test_covariance_matches_precomputed_means(sample_data):
    x, y = sample_data
    mx, my = minhastats.mean(x), minhastats.mean(y)
    assert math.isclose(minhastats.covariance(x, y, ddof=1), minhastats.covariance(x, y, ddof=1, _mx=mx, _my=my), abs_tol=1e-12)


class TestEdgeCases:
    def test_empty_list_functions(self):
        with pytest.raises(ValueError): minhastats.mean([])
        with pytest.raises(ValueError): minhastats.median([])
        with pytest.raises(ValueError): minhastats.mode([])
        with pytest.raises(ValueError): minhastats.amplitude([])
        with pytest.raises(ValueError): minhastats.quantile([], 0.5)
        with pytest.raises(ValueError): minhastats.variance([], ddof=0)

    def test_invalid_ddof(self):
        with pytest.raises(ValueError): minhastats.variance([1, 2, 3], ddof=2)
        with pytest.raises(ValueError): minhastats.covariance([1, 2, 3], [1, 2, 3], ddof=2)

    def test_insufficient_sample(self):
        with pytest.raises(ValueError): minhastats.variance([5.0], ddof=1)
        with pytest.raises(ValueError): minhastats.std_dev([5.0], ddof=1)
        with pytest.raises(ValueError): minhastats.covariance([1.0], [2.0], ddof=1)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError): minhastats.covariance([1, 2, 3], [1, 2])
        with pytest.raises(ValueError): minhastats.linear_regression([1, 2, 3], [1, 2])

    @pytest.mark.parametrize("invalid_q", [-0.1, 1.1, 2.0, -5.0])
    def test_quantile_invalid_q(self, invalid_q):
        with pytest.raises(ValueError): minhastats.quantile([1, 2, 3, 4, 5], invalid_q)

    @pytest.mark.parametrize("q,expected", [(0.0, 1), (0.5, 3), (1.0, 5)])
    def test_quantile_known_values(self, q, expected):
        assert math.isclose(minhastats.quantile([1, 2, 3, 4, 5], q), expected)

    def test_percentile_invalid_p(self):
        with pytest.raises(ValueError): minhastats.percentile([1, 2, 3], -1)
        with pytest.raises(ValueError): minhastats.percentile([1, 2, 3], 101)

    def test_cv_zero_mean(self):
        with pytest.raises(ValueError): minhastats.coefficient_of_variation([-1, 0, 1])

    def test_zero_variance_behavior(self):
        assert minhastats.pearson_correlation([5, 5, 5, 5], [1, 2, 3, 4]) == 0.0
        with pytest.raises(ValueError): minhastats.linear_regression([5, 5, 5, 5], [1, 2, 3, 4])
        b0, b1, r2 = minhastats.linear_regression([1, 2, 3, 4], [7, 7, 7, 7])
        assert math.isclose(b1, 0.0, abs_tol=1e-9)
        assert math.isclose(b0, 7.0, abs_tol=1e-9)
        assert r2 == 0.0


def test_median_known_values():
    assert minhastats.median([3, 1, 2]) == 2.0
    assert minhastats.median([1, 2, 3, 4]) == 2.5
