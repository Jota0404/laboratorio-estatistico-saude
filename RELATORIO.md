# 📊 Relatório Técnico de Desenvolvimento e Refatoração
## Laboratório Estatístico Interativo — Análise de Saúde


## 1. Visão Geral do Projeto

O **Laboratório Estatístico Interativo** é uma aplicação web interativa desenvolvida em Python e Streamlit com o objetivo de oferecer uma ferramenta didática e analítica para o processamento de dados de saúde pública e hospitalar. 

A aplicação aborda desde a análise descritiva inicial (medidas de tendência central, dispersão e detecção de *outliers*) até a modelagem inferencial e preditiva (distribuições de probabilidade, simulação de Monte Carlo via Lei dos Grandes Números e Regressão Linear Simples por Mínimos Quadrados).


## 2. Arquitetura e Decisões de Projeto

O projeto adota uma arquitetura modular baseada no princípio de **Responsabilidade Única (SRP)**:

- **Núcleo Estatístico (`src/minhastats.py`):** Implementado do zero em Python puro, sem dependências de frameworks externos para os cálculos fundamentais. Esse módulo é totalmente independente da camada de interface e de bibliotecas de manipulação de dados, garantindo portabilidade e facilidade de testes unitários.
- **Camada de Apresentação (`app.py`):** Utiliza o Streamlit para construir a interface do usuário, orquestrando as entradas do usuário, invocações do núcleo estatístico e renderização de dashboards interativos.
- **Suíte de Testes (`tests/test_minhastats.py`):** Conjunto de testes automatizados com `pytest` que validam a precisão matemática das implementações próprias contra resultados de referência de bibliotecas consolidadas (`numpy` e `scipy`).


## 3. Processo de Code Review e Refatoração

Após a etapa inicial de desenvolvimento, o código passou por uma revisão técnica detalhada (Code Review Sênior), resultando em melhorias críticas em quatro eixos principais:

### 3.1. Performance e Otimização Algorítmica
- **Vetorização da Lei dos Grandes Números:** A simulação acumulativa no Módulo 3 possuía um loop O(n²), recalculando somas consecutivas a cada passo. O trecho foi refatorado utilizando operações vetorizadas (`np.cumsum`), reduzindo a complexidade temporal para O(n) e eliminando travamentos na interface durante execuções com $n = 5.000$ amostras.
- **Eliminação de Recomputação Redundante:** O cálculo de regressão linear e covariância recalculava a média das variáveis múltiplas vezes. O núcleo em `minhastats.py` foi ajustado para reaproveitar parâmetros pré-calculados de média e variância em chamadas dependentes.

### 3.2. Robustez, Validação e Tratamento de Exceções
- **Tratamento na UI (`app.py`):** Foram adicionados blocos `try/except ValueError` em todas as abas interativas. Situações como amostragem insuficiente ($n \le \text{ddof}$), colunas com variância zero ou seleções nulas agora exibem alertas amigáveis (`st.warning`/`st.error`) com encerramento gracioso via `st.stop()`, evitando a exposição de tracebacks de erro ao usuário final.
- **Carregamento de Dados Determinístico:** A função de leitura de dados foi otimizada com `@st.cache_data` para evitar acessos repetidos ao disco, aplicando buscas ordenadas e tratamento específico para erros de parse do Pandas.

### 3.3. Testes Unitários e Cobertura de Edge Cases
A suíte de testes em `tests/test_minhastats.py` foi expandida para além do "caminho feliz", cobrindo:
- Validação de exceções (`pytest.raises`) para listas vazias, listas de tamanho ímpar/par e quantis fora do intervalo $[0, 1]$.
- Comportamento de exceção para variância zero e entradas com dimensões incompatíveis.

### 3.4. Segurança, Gestão de Memória e Boas Práticas
- **Conformidade LGPD / Segurança de Dados:** Atualização do `.gitignore` para garantir a exclusão de qualquer arquivo de dados reais de saúde (`data/raw/*.csv`), prevenindo a publicação acidental de dados sensíveis em repositórios públicos.
- **Gestão de Memória no Matplotlib:** Inclusão de `plt.close(fig)` em todas as renderizações de gráficos no Streamlit para evitar vazamento de memória em sessões prolongadas.
- **Reprodutibilidade:** Pinagem rigorosa de versões das dependências no `requirements.txt`.


## 4. Resultados e Status do Projeto

| Indicador | Status Inicial | Status Pós-Refatoração |
| :--- | :---: | :---: |
| **Complexidade da LGN** | $O(n^2)$ | **$O(n)$** |
| **Tratamento de Exceções na UI** | Inexistente (crashes expostos) | **Robusto (`st.error` / `st.stop`)** |
| **Cobertura de Testes (`pytest`)** | Caminho feliz apenas | **Edge cases + Exceções tratadas** |
| **Proteção de Dados (`.gitignore`)** | Desativada (risco LGPD) | **Ativa (`data/raw/*.csv` protegido)** |
| **Cache de Carregamento de Dados** | Ausente | **Ativo (`@st.cache_data`)** |


## 5. Próximos Passos (Trabalhos Futuros)

- **Módulo de Regressão Múltipla:** Expandir o núcleo estatístico para suportar mais de uma variável independente.
- **Exportação de Relatórios:** Permitir a geração automática de resumos executivos em PDF/HTML com os insights gerados na aba interativa.