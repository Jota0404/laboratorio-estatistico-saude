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
    assert math.isclose(minhastats.variance(x, ddof=1), np.var(x, ddof=1), abs_tol=tol)
    assert math.isclose(minhastats.std_dev(x, ddof=1), np.std(x, ddof=1), abs_tol=tol)
    assert math.isclose(minhastats.quantile(x, 0.25), np.percentile(x, 25), abs_tol=1e-2)

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