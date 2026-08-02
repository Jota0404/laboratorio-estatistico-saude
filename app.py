import glob

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src import minhastats

# Configuração da página
st.set_page_config(
    page_title="Laboratório Estatístico de Saúde",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Laboratório Estatístico Interativo — Análise de Saúde")
st.markdown("---")


# Carregamento dos dados
@st.cache_data
def load_data(data_dir: str = "data/raw"):
    """Carrega o primeiro arquivo CSV encontrado em `data_dir`.

    Args:
        data_dir: Diretório onde os arquivos CSV de dados estão
            armazenados.

    Returns:
        Um DataFrame com os dados carregados, ou `None` se nenhum
        arquivo CSV válido for encontrado.
    """
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not csv_files:
        st.error(f"Nenhum arquivo CSV encontrado em `{data_dir}/`.")
        return None

    path = csv_files[0]
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        st.error(f"Erro ao ler o arquivo '{path}': {e}")
        return None


df = load_data()

if df is not None:
    # Abas para navegação entre os Módulos do Trabalho
    tab0, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Módulo 0: Dados",
        "📊 Módulo 2: Descritiva Interativa",
        "🎲 Módulo 3: Monte Carlo (LGN & TCL)",
        "📐 Módulo 4: Distribuições Teóricas",
        "📈 Módulo 5: Correlação & Regressão"
    ])

    # ----------------------------------------------------
    # MÓDULO 0: DADOS REAIS
    # ----------------------------------------------------
    with tab0:
        st.header("📦 Visão Geral dos Dados de Saúde")
        st.dataframe(df.head(10), use_container_width=True)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total de Registros (Linhas)", len(df))
        col_m2.metric("Total de Colunas", len(df.columns))

    # Identificação automática de variáveis
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ----------------------------------------------------
    # MÓDULO 2: ESTATÍSTICA DESCRITIVA INTERATIVA
    # ----------------------------------------------------
    with tab2:
        st.header("📊 Análise Descritiva Univariada")
        var_sel = st.selectbox("Selecione uma Variável Numérica:", num_cols, key="mod2_var")

        data_clean = df[var_sel].dropna().tolist()

        if len(data_clean) < 2:
            st.warning(
                f"A variável **{var_sel}** possui menos de 2 valores válidos "
                "e não pode ser analisada estatisticamente."
            )
            st.stop()

        # Cálculos usando O NÚCLEO PRÓPRIO!
        try:
            mean_val = minhastats.mean(data_clean)
            med_val = minhastats.median(data_clean)
            std_val = minhastats.std_dev(data_clean, ddof=1)
            q1_val = minhastats.quantile(data_clean, 0.25)
            q3_val = minhastats.quantile(data_clean, 0.75)
        except ValueError as e:
            st.error(f"Não foi possível calcular as estatísticas de **{var_sel}**: {e}")
            st.stop()

        iqr_val = q3_val - q1_val

        # Detecção de Outliers via IQR
        lower_bound = q1_val - 1.5 * iqr_val
        upper_bound = q3_val + 1.5 * iqr_val
        outliers = [x for x in data_clean if x < lower_bound or x > upper_bound]

        # Métricas exibidas na tela
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média (Própria)", f"{mean_val:.2f}")
        c2.metric("Mediana (Própria)", f"{med_val:.2f}")
        c3.metric("Desvio Padrão (Próprio)", f"{std_val:.2f}")
        c4.metric("Outliers (Regra IQR)", len(outliers))

        # Gráficos
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data_clean, kde=True, ax=ax[0], color="skyblue")
        ax[0].set_title(f"Histograma & Densidade de {var_sel}")

        sns.boxplot(x=data_clean, ax=ax[1], color="lightgreen")
        ax[1].set_title(f"Boxplot de {var_sel}")
        st.pyplot(fig)
        plt.close(fig)

        # Interpretação Automática de Assimetria
        st.subheader("💡 Interpretação Automática")
        if abs(mean_val - med_val) < 0.05 * std_val:
            st.info("A distribuição aparenta ser aproximadamente **Simétrica** (Média próximo da Mediana).")
        elif mean_val > med_val:
            st.info("A distribuição apresenta **Assimetria Positiva (à direita)** (Média maior que a Mediana).")
        else:
            st.info("A distribuição apresenta **Assimetria Negativa (à esquerda)** (Média menor que a Mediana).")

    # ----------------------------------------------------
    # MÓDULO 3: PROBABILIDADE E SIMULAÇÃO DE MONTE CARLO
    # ----------------------------------------------------
    with tab3:
        st.header("🎲 Simulação de Monte Carlo")
        sim_choice = st.radio("Selecione o Experimento:", ["(a) Lei dos Grandes Números (LGN)", "(b) Teorema Central do Limite (TCL)"])

        if "(a)" in sim_choice:
            st.subheader("Lei dos Grandes Números — Lançamento de Dados Simulados")
            n_reps = st.slider("Número de Lançamentos:", 10, 5000, 500, step=50)

            rolls = np.random.randint(1, 7, size=n_reps)
            # Versão vetorizada com np.cumsum: O(n) em vez do loop original O(n²)
            # que recalculava a média de toda a lista a cada novo lançamento.
            cum_means = (np.cumsum(rolls) / np.arange(1, n_reps + 1)).tolist()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, n_reps + 1), cum_means, label="Média Acumulada", color="purple")
            ax.axhline(3.5, color="red", linestyle="--", label="Valor Esperado Teórico (3.5)")
            ax.set_xlabel("Número de Ensaios")
            ax.set_ylabel("Média Aritmética")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.subheader("Teorema Central do Limite (TCL)")
            var_tcl = st.selectbox("Escolha a variável do dataset:", num_cols, key="tcl_var")
            sample_size = st.slider("Tamanho da Amostra (n):", 5, 200, 30)
            num_samples = st.slider("Número de Amostras Simuladas:", 100, 5000, 1000)

            pop_data = df[var_tcl].dropna().tolist()

            if len(pop_data) < sample_size:
                st.warning(
                    f"A variável **{var_tcl}** possui apenas {len(pop_data)} valores válidos, "
                    f"insuficiente para amostras de tamanho {sample_size}."
                )
                st.stop()

            try:
                sample_means = []
                for _ in range(num_samples):
                    sample = np.random.choice(pop_data, size=sample_size, replace=True)
                    sample_means.append(minhastats.mean(sample.tolist()))
            except ValueError as e:
                st.error(f"Não foi possível simular as amostras de **{var_tcl}**: {e}")
                st.stop()

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(sample_means, kde=True, ax=ax, color="orange")
            ax.set_title(f"Distribuição das Médias Amostrais (n={sample_size}, k={num_samples})")
            st.pyplot(fig)
            plt.close(fig)

    # ----------------------------------------------------
    # MÓDULO 4: DISTRIBUIÇÕES TEÓRICAS
    # ----------------------------------------------------
    with tab4:
        st.header("📐 Ajuste de Distribuições Teóricas")
        var_dist = st.selectbox("Escolha a variável para ajustar:", num_cols, key="dist_var")
        dist_type = st.selectbox("Escolha a Distribuição Candidata:", ["Normal", "Exponencial"])

        data_dist = df[var_dist].dropna().tolist()

        if len(data_dist) < 2:
            st.warning(
                f"A variável **{var_dist}** possui menos de 2 valores válidos "
                "e não pode ter uma distribuição ajustada."
            )
            st.stop()

        try:
            m_hat = minhastats.mean(data_dist)
            s_hat = minhastats.std_dev(data_dist, ddof=1)
        except ValueError as e:
            st.error(f"Não foi possível ajustar a distribuição de **{var_dist}**: {e}")
            st.stop()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data_dist, stat="density", ax=ax, color="gray", label="Dados Reais")

        x_vals = np.linspace(min(data_dist), max(data_dist), 200)

        if dist_type == "Normal":
            pdf = stats.norm.pdf(x_vals, loc=m_hat, scale=s_hat)
            ax.plot(x_vals, pdf, 'r-', lw=2, label=f"Ajuste Normal (μ={m_hat:.1f}, σ={s_hat:.1f})")
        else:
            if min(data_dist) < 0:
                st.warning(
                    f"A variável **{var_dist}** possui valores negativos; "
                    "a distribuição Exponencial só é definida para valores ≥ 0. "
                    "O ajuste abaixo pode não ser representativo."
                )
            pdf = stats.expon.pdf(x_vals, scale=m_hat)
            ax.plot(x_vals, pdf, 'g-', lw=2, label=f"Ajuste Exponencial (λ=1/{m_hat:.1f})")

        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ----------------------------------------------------
    # MÓDULO 5: CORRELAÇÃO E REGRESSÃO LINEAR
    # ----------------------------------------------------
    with tab5:
        st.header("📈 Correlação e Regressão Linear Simples")
        col_x, col_y = st.columns(2)
        x_var = col_x.selectbox("Variável Independente (X):", num_cols, index=0)
        y_var = col_y.selectbox("Variável Dependente (Y):", num_cols, index=min(1, len(num_cols)-1))

        df_reg = df[[x_var, y_var]].dropna()
        x_list = df_reg[x_var].tolist()
        y_list = df_reg[y_var].tolist()

        if len(x_list) < 2:
            st.warning(
                f"Não há pares de valores suficientes entre **{x_var}** e **{y_var}** "
                "para ajustar um modelo de regressão."
            )
            st.stop()

        # Cálculos via Núcleo Próprio!
        try:
            b0, b1, r2 = minhastats.linear_regression(x_list, y_list)
            r = minhastats.pearson_correlation(x_list, y_list)
        except ValueError as e:
            st.error(
                f"⚠️ Não foi possível ajustar o modelo entre **{x_var}** e **{y_var}**: {e}. "
                f"Verifique se a variável '{x_var}' possui variação nos dados."
            )
            st.stop()

        st.subheader("Resultados do Modelo (Mínimos Quadrados Próprios)")
        st.latex(fr"\hat{{Y}} = {b0:.4f} + {b1:.4f} \cdot X")

        m1, m2 = st.columns(2)
        m1.metric("Coeficiente de Correlação (r)", f"{r:.4f}")
        m2.metric("Coeficiente de Determinação (R²)", f"{r2:.4f}")

        # Predição Interativa
        st.markdown("---")
        st.subheader("🔮 Campo de Predição Interativa")
        x_pred = st.number_input(f"Digite um valor para {x_var} (X):", value=float(minhastats.mean(x_list)))
        y_hat = b0 + b1 * x_pred
        st.success(rf"**Valor Predito para {y_var} ($\hat{{Y}}$):** {y_hat:.2f}")

        # Gráfico de Dispersão com Reta
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(x_list, y_list, alpha=0.3, label="Pacientes / Dados Reais")
        x_range = np.linspace(min(x_list), max(x_list), 100)
        ax.plot(x_range, b0 + b1 * x_range, color='red', linewidth=2, label="Reta de Regressão OLS")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.warning("⚠️ **Alerta:** Correlação estatística não implica causalidade clínica!")