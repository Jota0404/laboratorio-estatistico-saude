import math

def mean(data: list[float]) -> float:
    """Média Aritmética: sum(x) / n"""
    if not data:
        raise ValueError("A lista de dados não pode estar vazia.")
    return sum(data) / len(data)

def median(data: list[float]) -> float:
    """Mediana: valor central após ordenação"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        raise ValueError("A lista de dados não pode estar vazia.")
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    return float(sorted_data[mid])

def variance(data: list[float], ddof: int = 1) -> float:
    """Variância: ddof=1 (amostral) ou ddof=0 (populacional)"""
    n = len(data)
    if n <= ddof:
        raise ValueError("Dados insuficientes para o ddof especificado.")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)

def std_dev(data: list[float], ddof: int = 1) -> float:
    """Desvio Padrão"""
    return math.sqrt(variance(data, ddof=ddof))

def quantile(data: list[float], q: float) -> float:
    """Percentil/Quartil via interpolação linear (q entre 0 e 1)"""
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

def covariance(x: list[float], y: list[float], ddof: int = 1) -> float:
    """Covariância entre X e Y"""
    if len(x) != len(y) or len(x) <= ddof:
        raise ValueError("Listas devem ter o mesmo tamanho e n > ddof.")
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - ddof)

def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Coeficiente de Correlação de Pearson"""
    sx, sy = std_dev(x, ddof=1), std_dev(y, ddof=1)
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y, ddof=1) / (sx * sy)

def linear_regression(x: list[float], y: list[float]):
    """
    Regressão Linear Simples via Mínimos Quadrados (OLS)
    Retorna: beta_0 (intercepto), beta_1 (inclinação), r2 (coeficiente de determinação)
    """
    var_x = variance(x, ddof=1)
    if var_x == 0:
        raise ValueError("Variância de X não pode ser zero.")
    beta_1 = covariance(x, y, ddof=1) / var_x
    beta_0 = mean(y) - beta_1 * mean(x)
    r = pearson_correlation(x, y)
    return beta_0, beta_1, r ** 2 