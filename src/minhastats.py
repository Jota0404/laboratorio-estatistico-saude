import math
from typing import Optional


def mean(data: list[float]) -> float:
    """Calcula a média aritmética de uma lista de valores.

    Args:
        data: Lista de valores numéricos.

    Returns:
        A média aritmética: sum(x) / n.

    Raises:
        ValueError: Se `data` estiver vazia.
    """
    if not data:
        raise ValueError("A lista de dados não pode estar vazia.")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calcula a mediana de uma lista de valores.

    Args:
        data: Lista de valores numéricos.

    Returns:
        O valor central após ordenação. Para `n` par, retorna a média
        dos dois valores centrais.

    Raises:
        ValueError: Se `data` estiver vazia.
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        raise ValueError("A lista de dados não pode estar vazia.")
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    return float(sorted_data[mid])


def variance(
    data: list[float],
    ddof: int = 1,
    _mean: Optional[float] = None,
) -> float:
    """Calcula a variância de uma lista de valores.

    Args:
        data: Lista de valores numéricos.
        ddof: Graus de liberdade (delta degrees of freedom).
            Use `ddof=1` para variância amostral (padrão) ou `ddof=0`
            para variância populacional.
        _mean: Média pré-calculada de `data`, opcional. Quando fornecida,
            evita recalcular `mean(data)` — útil para funções que já
            possuem esse valor (ex. `covariance`, `linear_regression`),
            eliminando uma passada O(n) redundante sobre os dados.

    Returns:
        A variância de `data`.

    Raises:
        ValueError: Se `len(data) <= ddof` (dados insuficientes para o
            grau de liberdade especificado).
    """
    n = len(data)
    if n <= ddof:
        raise ValueError("Dados insuficientes para o ddof especificado.")
    m = _mean if _mean is not None else mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)


def std_dev(data: list[float], ddof: int = 1) -> float:
    """Calcula o desvio padrão de uma lista de valores.

    Args:
        data: Lista de valores numéricos.
        ddof: Graus de liberdade. `ddof=1` (padrão) para desvio padrão
            amostral, `ddof=0` para populacional.

    Returns:
        O desvio padrão: sqrt(variance(data, ddof)).

    Raises:
        ValueError: Se `len(data) <= ddof`.
    """
    return math.sqrt(variance(data, ddof=ddof))


def quantile(data: list[float], q: float) -> float:
    """Calcula um percentil/quartil via interpolação linear.

    Args:
        data: Lista de valores numéricos.
        q: Quantil desejado, entre 0 e 1 (ex. 0.25 para o primeiro
            quartil, 0.5 para a mediana, 0.75 para o terceiro quartil).

    Returns:
        O valor interpolado correspondente ao quantil `q`.

    Raises:
        ValueError: Se `data` estiver vazia ou se `q` não estiver no
            intervalo [0, 1].
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0 or not (0 <= q <= 1):
        raise ValueError("q deve estar entre 0 e 1.")
    pos = q * (n - 1)
    base = math.floor(pos)
    rest = pos - base
    if base + 1 < n:
        return sorted_data[base] + rest * (sorted_data[base + 1] - sorted_data[base])
    return float(sorted_data[base])


def covariance(
    x: list[float],
    y: list[float],
    ddof: int = 1,
    _mx: Optional[float] = None,
    _my: Optional[float] = None,
) -> float:
    """Calcula a covariância amostral ou populacional entre X e Y.

    Args:
        x: Lista de valores numéricos da primeira variável.
        y: Lista de valores numéricos da segunda variável, mesmo
            tamanho de `x`.
        ddof: Graus de liberdade. `ddof=1` (padrão) para covariância
            amostral, `ddof=0` para populacional.
        _mx: Média pré-calculada de `x`, opcional (evita recomputação).
        _my: Média pré-calculada de `y`, opcional (evita recomputação).

    Returns:
        A covariância entre `x` e `y`.

    Raises:
        ValueError: Se `len(x) != len(y)` ou se `len(x) <= ddof`.
    """
    if len(x) != len(y) or len(x) <= ddof:
        raise ValueError("Listas devem ter o mesmo tamanho e n > ddof.")
    mx = _mx if _mx is not None else mean(x)
    my = _my if _my is not None else mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - ddof)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Calcula o coeficiente de correlação de Pearson entre X e Y.

    Reaproveita `mean(x)` e `mean(y)` entre as chamadas internas de
    `variance` e `covariance`, evitando recomputações O(n) redundantes.

    Args:
        x: Lista de valores numéricos da primeira variável.
        y: Lista de valores numéricos da segunda variável, mesmo
            tamanho de `x`.

    Returns:
        O coeficiente de correlação de Pearson (entre -1 e 1). Retorna
        `0.0` se `x` ou `y` tiverem variância zero (correlação
        matematicamente indefinida nesse caso).

    Raises:
        ValueError: Se `len(x) != len(y)` ou se houver menos de 2
            observações.
    """
    mx, my = mean(x), mean(y)
    var_x = variance(x, ddof=1, _mean=mx)
    var_y = variance(y, ddof=1, _mean=my)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov_xy = covariance(x, y, ddof=1, _mx=mx, _my=my)
    return cov_xy / math.sqrt(var_x * var_y)


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Ajusta uma regressão linear simples via Mínimos Quadrados (OLS).

    Calcula `mean(x)` e `mean(y)` uma única vez e reaproveita esses
    valores em `variance` e `covariance`, evitando as recomputações
    O(n) redundantes que existiam ao encadear `variance`,
    `covariance` e `pearson_correlation` de forma independente.

    Args:
        x: Lista de valores da variável independente.
        y: Lista de valores da variável dependente, mesmo tamanho de
            `x`.

    Returns:
        Uma tupla `(beta_0, beta_1, r2)`, onde:
            beta_0: Intercepto da reta ajustada.
            beta_1: Inclinação (coeficiente angular) da reta ajustada.
            r2: Coeficiente de determinação (R²).

    Raises:
        ValueError: Se a variância de `x` for zero (reta indefinida)
            ou se `len(x) != len(y)`.
    """
    mx, my = mean(x), mean(y)
    var_x = variance(x, ddof=1, _mean=mx)
    if var_x == 0:
        raise ValueError("Variância de X não pode ser zero.")
    var_y = variance(y, ddof=1, _mean=my)
    cov_xy = covariance(x, y, ddof=1, _mx=mx, _my=my)

    beta_1 = cov_xy / var_x
    beta_0 = my - beta_1 * mx
    r = 0.0 if var_y == 0 else cov_xy / math.sqrt(var_x * var_y)
    return beta_0, beta_1, r ** 2