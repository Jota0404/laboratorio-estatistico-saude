# Relatório — Laboratório Estatístico Interativo de Saúde

## 1. Identificação

- **Aluno:** João Marcos de Barcelos Fernandes
- **Matrícula:** 72650311
- **Disciplina:** [Matemática e Estatística para Computação](https://campusonline1.ceub.br/course/view.php?id=2382)
- **Repositório:** https://github.com/Jota0404/laboratorio-estatistico-saude

## 2. Objetivo

O projeto consiste na construção de um laboratório estatístico interativo em Python e Streamlit, aplicado a um dataset real da área da saúde. A aplicação permite explorar estatística descritiva, simulações de Monte Carlo, distribuições teóricas, correlação e regressão linear, mantendo os cálculos estatísticos fundamentais em uma biblioteca própria.

A arquitetura separa a interface (`app.py`), o núcleo matemático (`src/minhastats.py`) e os testes automatizados (`tests/test_minhastats.py`).

## 3. Dataset escolhido

Foi escolhido o dataset **Diabetes 130-US Hospitals for Years 1999-2008**, disponível no UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008

O conjunto possui 101.766 registros e 47 atributos relacionados a atendimentos hospitalares de pacientes com diabetes. Ele oferece variáveis numéricas e categóricas suficientes para exploração estatística e análise das relações entre características do atendimento e resultados clínicos.

O download é reproduzível por meio de:

```bash
python scripts/download_dataset.py
```

Os dados brutos não são versionados no Git. O script baixa o pacote diretamente da fonte pública e extrai `diabetic_data.csv` para `data/raw/`.

## 4. Arquitetura e decisões de implementação

### 4.1 Núcleo estatístico

O arquivo `src/minhastats.py` foi desenvolvido em Python puro. As funções fundamentais não delegam os cálculos para NumPy ou SciPy.

São implementados:

- média;
- mediana;
- moda;
- amplitude;
- variância populacional;
- variância amostral;
- desvio padrão populacional;
- desvio padrão amostral;
- quartis e percentis;
- coeficiente de variação;
- covariância;
- correlação de Pearson;
- regressão linear simples por mínimos quadrados.

### 4.2 Interface

`app.py` utiliza Streamlit para permitir seleção de variáveis, alteração de parâmetros de simulação, visualização de tabelas e gráficos e interpretação automática.

### 4.3 Testes

A suíte `pytest` compara os cálculos próprios com NumPy/SciPy e cobre casos de borda, como listas vazias, dados insuficientes, tamanhos incompatíveis, quantis inválidos e variância zero.

## 5. Fórmulas utilizadas

### 5.1 Média

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

### 5.2 Variância populacional

$$
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

### 5.3 Variância amostral

$$
 s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

### 5.4 Desvio padrão

$$
\sigma=\sqrt{\sigma^2}, \qquad s=\sqrt{s^2}
$$

### 5.5 Amplitude

$$
A=\max(x)-\min(x)
$$

### 5.6 Coeficiente de variação

$$
CV=\frac{s}{|\bar{x}|}\times100
$$

### 5.7 Covariância amostral

$$
Cov(X,Y)=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
$$

### 5.8 Correlação de Pearson

$$
r=\frac{Cov(X,Y)}{s_Xs_Y}
$$

### 5.9 Regressão linear simples

$$
\hat{Y}=\beta_0+\beta_1X
$$

com

$$
\beta_1=\frac{Cov(X,Y)}{Var(X)}
$$

$$
\beta_0=\bar{Y}-\beta_1\bar{X}
$$

$$
R^2=r^2
$$

## 6. Validação dos cálculos

Os resultados do núcleo próprio são comparados com implementações consolidadas durante os testes automatizados:

| Função própria | Referência |
|---|---|
| Média | `numpy.mean` |
| Mediana | `numpy.median` |
| Variância | `numpy.var` |
| Desvio padrão | `numpy.std` |
| Percentil | `numpy.percentile` |
| Covariância | `numpy.cov` |
| Pearson | `scipy.stats.pearsonr` |
| Regressão | `scipy.stats.linregress` |

Além da comparação numérica, os testes verificam comportamentos de exceção e casos extremos.

Executar:

```bash
pytest
```

## 7. Módulo 0 — Dados

O primeiro módulo apresenta uma amostra do dataset, total de registros, quantidade de variáveis numéricas e categóricas, tipos de dados e quantidade de valores ausentes.

**Evidência a inserir na versão final:** screenshot do Módulo 0 em execução.

## 8. Módulo 2 — Estatística Descritiva Interativa

Para variáveis numéricas, a aplicação apresenta:

- média;
- mediana;
- moda;
- amplitude;
- variância populacional e amostral;
- desvio padrão populacional e amostral;
- quartis;
- percentil 90;
- coeficiente de variação;
- tabela de frequência absoluta e relativa;
- histograma;
- boxplot;
- detecção de outliers pela regra do IQR;
- interpretação automática da assimetria e da dispersão.

Para variáveis categóricas, apresenta tabela de frequências e gráfico de barras das categorias mais frequentes.

**Evidência a inserir na versão final:** screenshot do módulo mostrando métricas, tabela e gráficos.

## 9. Módulo 3 — Monte Carlo

### 9.1 Lei dos Grandes Números

É simulada uma sequência de lançamentos de um dado justo. A média acumulada é comparada ao valor esperado teórico 3,5. O número de lançamentos e a semente aleatória podem ser alterados pelo usuário.

A média acumulada é calculada de forma vetorizada com `numpy.cumsum`, evitando recomputação quadrática.

### 9.2 Teorema Central do Limite

A aplicação seleciona uma variável numérica do dataset e gera várias amostras com reposição. O histograma das médias amostrais permite observar a tendência de aproximação à distribuição Normal conforme o tamanho da amostra aumenta.

**Evidência a inserir na versão final:** screenshots da LGN e do TCL.

## 10. Módulo 4 — Distribuições Teóricas

São disponibilizadas duas distribuições candidatas:

- Normal;
- Exponencial.

Os parâmetros da Normal são estimados a partir da média e do desvio padrão do dataset. Para a Exponencial, a escala é estimada pela média, com validação de domínio para impedir ajustes inválidos em dados negativos.

As curvas são sobrepostas ao histograma dos dados reais. A aplicação também apresenta uma interpretação textual da adequação esperada do ajuste.

Durante a validação manual, a variável `time_in_hospital` apresentou comportamento claramente assimétrico. No ajuste Normal, a assimetria amostral observada foi **1,134**, com evidente afastamento entre o histograma e a curva Normal. No ajuste Exponencial, a curva acompanhou a tendência geral de decaimento, mas os picos discretos dos valores inteiros permaneceram afastados da curva estimada. Esses resultados mostram que a adequação da distribuição deve ser avaliada visualmente, sem presumir que uma distribuição teórica necessariamente represente bem os dados.

**Evidência a inserir na versão final:** screenshot dos dois ajustes.

## 11. Módulo 5 — Correlação e Regressão

O usuário seleciona duas variáveis numéricas. O módulo apresenta:

- coeficiente de Pearson;
- equação da reta de regressão;
- R²;
- gráfico de dispersão;
- reta de mínimos quadrados;
- campo de predição interativa;
- interpretação da força da associação;
- alerta explícito de que correlação não implica causalidade.

Na validação manual com `time_in_hospital` como X e `num_medications` como Y, foi obtido **r = 0,4661** e **R² = 0,2173**, caracterizando associação linear moderada e indicando que o modelo explica aproximadamente 21,73% da variação observada em `num_medications`.

**Evidência a inserir na versão final:** screenshot do módulo com equação, R² e gráfico.

## 12. Módulo 6 — Três descobertas estatísticas

O módulo apresenta automaticamente três descobertas sobre a variável selecionada:

### Descoberta 1 — Tendência central

Compara média e mediana e identifica a direção da assimetria observada.

### Descoberta 2 — Dispersão

Relaciona o desvio padrão ao coeficiente de variação para caracterizar a variabilidade relativa.

### Descoberta 3 — Valores extremos

Quantifica os outliers identificados pela regra do IQR e informa sua proporção no conjunto analisado.

### Registro final das descobertas

Para a variável **`time_in_hospital`**, foram observados os seguintes resultados na execução real do aplicativo:

- **Descoberta final 1 — Tendência central:** a média foi **4,40** e a mediana **4,00**. Como a média é maior que a mediana, os dados apresentam **assimetria positiva**.
- **Descoberta final 2 — Dispersão:** o desvio padrão amostral foi **2,99** e o coeficiente de variação foi **67,91%**, indicando **elevada variabilidade relativa** em torno da média.
- **Descoberta final 3 — Valores extremos:** a regra do IQR identificou **2.252 outliers** entre **101.766 observações**, correspondendo a aproximadamente **2,21%** do conjunto.

Esses valores foram registrados a partir da execução real do módulo com `time_in_hospital` selecionada e devem ser acompanhados pela captura de tela correspondente na versão final entregue.

## 13. Conclusão

O laboratório integra programação, estatística e visualização de dados em uma aplicação interativa. A separação entre núcleo matemático, interface e testes permite validar os cálculos próprios independentemente da camada visual e torna o projeto mais fácil de manter e reproduzir.

A etapa final da entrega deve incluir as evidências visuais reais da aplicação, a identificação do aluno e, preferencialmente, a consolidação das três descobertas estatísticas registradas neste relatório. O vídeo demonstrativo permanece como item não produzido nesta entrega.

## 14. Referências

- UCI Machine Learning Repository. **Diabetes 130-US Hospitals for Years 1999-2008**. https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008
- NumPy Documentation. https://numpy.org/doc/
- SciPy Documentation. https://docs.scipy.org/doc/scipy/
- Streamlit Documentation. https://docs.streamlit.io/
