import math
from collections import Counter
from typing import Optional


def _validate_data(data: list[float]) -> None:
    if not data:
        raise ValueError("A lista de dados não pode estar vazia.")


def mean(data: list[float]) -> float:
    """Calcula a média aritmética: sum(x) / n."""
    _validate_data(data)
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calcula a mediana, usando a média dos dois valores centrais quando n é par."""
    _validate_data(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    return float(sorted_data[mid])


def mode(data: list[float]) -> list[float]:
    """Retorna todas as modas, isto é, os valores com maior frequência."""
    _validate_data(data)
    counts = Counter(data)
    max_count = max(counts.values())
    return sorted([value for value, count in counts.items() if count == max_count])


def amplitude(data: list[float]) -> float:
    """Calcula a amplitude total: máximo - mínimo."""
    _validate_data(data)
    return max(data) - min(data)


def variance(
    data: list[float],
    ddof: int = 1,
    _mean: Optional[float] = None,
) -> float:
    """Calcula variância amostral (ddof=1) ou populacional (ddof=0)."""
    _validate_data(data)
    n = len(data)
    if ddof not in (0, 1):
        raise ValueError("ddof deve ser 0 (populacional) ou 1 (amostral).")
    if n <= ddof:
        raise ValueError("Dados insuficientes para o ddof especificado.")
    m = _mean if _mean is not None else mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)


def std_dev(data: list[float], ddof: int = 1) -> float:
    """Calcula desvio padrão amostral (ddof=1) ou populacional (ddof=0)."""
    return math.sqrt(variance(data, ddof=ddof))


def quantile(data: list[float], q: float) -> float:
    """Calcula um quantil/percentil por interpolação linear."""
    _validate_data(data)
    if not 0 <= q <= 1:
        raise ValueError("q deve estar entre 0 e 1.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    pos = q * (n - 1)
    base = math.floor(pos)
    rest = pos - base
    if base + 1 < n:
        return sorted_data[base] + rest * (sorted_data[base + 1] - sorted_data[base])
    return float(sorted_data[base])


def percentile(data: list[float], p: float) -> float:
    """Calcula um percentil em escala de 0 a 100."""
    if not 0 <= p <= 100:
        raise ValueError("p deve estar entre 0 e 100.")
    return quantile(data, p / 100.0)


def coefficient_of_variation(data: list[float], ddof: int = 1) -> float:
    """Calcula o coeficiente de variação em porcentagem.

    CV = (desvio padrão / |média|) * 100.
    """
    m = mean(data)
    if m == 0:
        raise ValueError("Coeficiente de variação indefinido quando a média é zero.")
    return std_dev(data, ddof=ddof) / abs(m) * 100.0


def covariance(
    x: list[float],
    y: list[float],
    ddof: int = 1,
    _mx: Optional[float] = None,
    _my: Optional[float] = None,
) -> float:
    """Calcula a covariância amostral ou populacional entre X e Y."""
    if len(x) != len(y):
        raise ValueError("Listas devem ter o mesmo tamanho.")
    if not x:
        raise ValueError("As listas não podem estar vazias.")
    if ddof not in (0, 1):
        raise ValueError("ddof deve ser 0 (populacional) ou 1 (amostral).")
    if len(x) <= ddof:
        raise ValueError("Dados insuficientes para o ddof especificado.")
    mx = _mx if _mx is not None else mean(x)
    my = _my if _my is not None else mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - ddof)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Calcula o coeficiente de correlação linear de Pearson."""
    if len(x) != len(y):
        raise ValueError("Listas devem ter o mesmo tamanho.")
    mx, my = mean(x), mean(y)
    var_x = variance(x, ddof=1, _mean=mx)
    var_y = variance(y, ddof=1, _mean=my)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov_xy = covariance(x, y, ddof=1, _mx=mx, _my=my)
    return cov_xy / math.sqrt(var_x * var_y)


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Ajusta regressão linear simples por Mínimos Quadrados (OLS).

    Retorna (intercepto, inclinação, R²).
    """
    if len(x) != len(y):
        raise ValueError("Listas devem ter o mesmo tamanho.")
    mx, my = mean(x), mean(y)
    var_x = variance(x, ddof=1, _mean=mx)
    if var_x == 0:
        raise ValueError("Variância de X não pode ser zero.")
    var_y = variance(y, ddof=1, _mean=my)
    cov_xy = covariance(x, y, ddof=1, _mx=mx, _my=my)

    beta_1 = cov_xy / var_x
    beta_0 = my - beta_1 * mx
    r = 0.0 if var_y == 0 else cov_xy / math.sqrt(var_x * var_y)
    return beta_0, beta_1, r**2
