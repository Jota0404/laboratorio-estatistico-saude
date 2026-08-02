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


def test_variance_matches_precomputed_mean(sample_data):
    """Garante que passar `_mean` já calculado não altera o resultado."""
    x, _ = sample_data
    m = minhastats.mean(x)
    assert math.isclose(
        minhastats.variance(x, ddof=1),
        minhastats.variance(x, ddof=1, _mean=m),
        abs_tol=1e-12,
    )


def test_covariance_matches_precomputed_means(sample_data):
    """Garante que passar `_mx`/`_my` já calculados não altera o resultado."""
    x, y = sample_data
    mx, my = minhastats.mean(x), minhastats.mean(y)
    assert math.isclose(
        minhastats.covariance(x, y, ddof=1),
        minhastats.covariance(x, y, ddof=1, _mx=mx, _my=my),
        abs_tol=1e-12,
    )


class TestEmptyAndInsufficientData:
    """Casos de borda: listas vazias ou com dados insuficientes."""

    def test_mean_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            minhastats.mean([])

    def test_median_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            minhastats.median([])

    def test_quantile_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            minhastats.quantile([], 0.5)

    def test_variance_raises_when_n_leq_ddof(self):
        # n=1, ddof=1 -> variância amostral indefinida
        with pytest.raises(ValueError):
            minhastats.variance([5.0], ddof=1)

    def test_variance_raises_when_data_empty_and_ddof_zero(self):
        # n=0, ddof=0 -> ainda insuficiente (0 <= 0)
        with pytest.raises(ValueError):
            minhastats.variance([], ddof=0)

    def test_std_dev_raises_when_n_leq_ddof(self):
        with pytest.raises(ValueError):
            minhastats.std_dev([5.0], ddof=1)

    def test_covariance_raises_when_n_leq_ddof(self):
        with pytest.raises(ValueError):
            minhastats.covariance([1.0], [2.0], ddof=1)


class TestMismatchedLengths:
    """Casos de borda: listas x/y com tamanhos incompatíveis."""

    def test_covariance_raises_on_mismatched_length(self):
        with pytest.raises(ValueError):
            minhastats.covariance([1, 2, 3], [1, 2])

    def test_linear_regression_raises_on_mismatched_length(self):
        # variância de x válida, mas covariância falha por tamanho incompatível
        with pytest.raises(ValueError):
            minhastats.linear_regression([1, 2, 3, 4], [1, 2, 3])


class TestQuantileInvalidQ:
    """Casos de borda: valores de q fora do intervalo [0, 1]."""

    @pytest.mark.parametrize("invalid_q", [-0.1, 1.1, 2.0, -5.0])
    def test_quantile_raises_on_invalid_q(self, invalid_q):
        with pytest.raises(ValueError):
            minhastats.quantile([1, 2, 3, 4, 5], invalid_q)

    @pytest.mark.parametrize("q,expected", [(0.0, 1), (0.5, 3), (1.0, 5)])
    def test_quantile_known_values(self, q, expected):
        data = [1, 2, 3, 4, 5]
        assert math.isclose(minhastats.quantile(data, q), expected)


class TestMedianKnownValues:
    """Valores exatos conhecidos para garantir cobertura dos dois ramos (par/ímpar)."""

    def test_median_odd_length(self):
        assert minhastats.median([3, 1, 2]) == 2.0

    def test_median_even_length(self):
        assert minhastats.median([1, 2, 3, 4]) == 2.5


class TestZeroVarianceBehavior:
    """Documenta o comportamento esperado quando a variância é zero."""

    def test_pearson_correlation_zero_variance_returns_zero(self):
        x = [5, 5, 5, 5]
        y = [1, 2, 3, 4]
        assert minhastats.pearson_correlation(x, y) == 0.0

    def test_linear_regression_zero_variance_raises(self):
        x = [5, 5, 5, 5]
        y = [1, 2, 3, 4]
        with pytest.raises(ValueError):
            minhastats.linear_regression(x, y)

    def test_linear_regression_zero_variance_in_y_does_not_raise(self):
        # Variância de X é válida; variância de Y zero não deve
        # impedir o ajuste (reta horizontal), apenas zera o r².
        x = [1, 2, 3, 4]
        y = [7, 7, 7, 7]
        b0, b1, r2 = minhastats.linear_regression(x, y)
        assert math.isclose(b1, 0.0, abs_tol=1e-9)
        assert math.isclose(b0, 7.0, abs_tol=1e-9)
        assert r2 == 0.0