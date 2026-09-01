import glob
import math

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src import minhastats

st.set_page_config(
    page_title="Laboratório Estatístico de Saúde",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Laboratório Estatístico Interativo — Análise de Saúde")
st.markdown("---")


def render_figure(fig):
    """Renderiza uma figura Matplotlib ocupando responsivamente o container."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data
def load_data(data_dir: str = "data/raw"):
    """Carrega o primeiro CSV local disponível."""
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not csv_files:
        return None
    path = csv_files[0]
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        st.error(f"Erro ao ler o arquivo '{path}': {exc}")
        return None


df = load_data()

if df is None:
    st.error("Nenhum CSV foi encontrado em `data/raw/`.")
    st.info(
        "Baixe o dataset público indicado no README e salve o CSV em `data/raw/`. "
        "O projeto não versiona dados brutos automaticamente."
    )
    st.stop()

if df.empty:
    st.error("O dataset está vazio.")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if len(num_cols) == 0:
    st.error("O dataset precisa possuir pelo menos uma variável numérica.")
    st.stop()

tab0, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📦 Módulo 0: Dados",
    "📊 Módulo 2: Descritiva Interativa",
    "🎲 Módulo 3: Monte Carlo (LGN & TCL)",
    "📐 Módulo 4: Distribuições Teóricas",
    "📈 Módulo 5: Correlação & Regressão",
    "🔎 Módulo 6: Descobertas",
])

with tab0:
    st.header("📦 Visão Geral dos Dados de Saúde")
    st.dataframe(df.head(10), use_container_width=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total de Registros", len(df))
    col_m2.metric("Variáveis Numéricas", len(num_cols))
    col_m3.metric("Variáveis Categóricas", len(cat_cols))

    st.subheader("Tipos de variáveis")
    type_table = pd.DataFrame({
        "Variável": df.columns,
        "Tipo": [str(df[column].dtype) for column in df.columns],
        "Nulos": [int(df[column].isna().sum()) for column in df.columns],
    })
    st.dataframe(type_table, use_container_width=True, hide_index=True)

with tab2:
    st.header("📊 Análise Descritiva Interativa")
    st.caption("As estatísticas principais são calculadas pelo núcleo próprio em `src/minhastats.py`.")
    variable_type = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)

    if variable_type == "Numérica":
        var_sel = st.selectbox("Selecione uma variável numérica:", num_cols, key="mod2_var")
        data_clean = df[var_sel].dropna().astype(float).tolist()
        if len(data_clean) < 2:
            st.warning(f"A variável **{var_sel}** possui menos de 2 valores válidos.")
            st.stop()
        try:
            mean_val = minhastats.mean(data_clean)
            med_val = minhastats.median(data_clean)
            modes = minhastats.mode(data_clean)
            amp_val = minhastats.amplitude(data_clean)
            var_pop = minhastats.variance(data_clean, ddof=0)
            var_sample = minhastats.variance(data_clean, ddof=1)
            std_pop = minhastats.std_dev(data_clean, ddof=0)
            std_sample = minhastats.std_dev(data_clean, ddof=1)
            q1_val = minhastats.quantile(data_clean, 0.25)
            q2_val = minhastats.quantile(data_clean, 0.50)
            q3_val = minhastats.quantile(data_clean, 0.75)
            p90_val = minhastats.percentile(data_clean, 90)
            cv_val = minhastats.coefficient_of_variation(data_clean)
        except ValueError as exc:
            st.error(f"Não foi possível calcular as estatísticas: {exc}")
            st.stop()

        iqr_val = q3_val - q1_val
        lower_bound = q1_val - 1.5 * iqr_val
        upper_bound = q3_val + 1.5 * iqr_val
        outliers = [x for x in data_clean if x < lower_bound or x > upper_bound]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média", f"{mean_val:.2f}")
        c2.metric("Mediana", f"{med_val:.2f}")
        c3.metric("Moda", ", ".join(f"{value:.2f}" for value in modes[:3]))
        c4.metric("Outliers (IQR)", len(outliers))
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Amplitude", f"{amp_val:.2f}")
        c6.metric("Variância amostral", f"{var_sample:.2f}")
        c7.metric("Desvio padrão amostral", f"{std_sample:.2f}")
        c8.metric("CV", f"{cv_val:.2f}%")

        st.subheader("Variância e desvio padrão")
        dispersion_table = pd.DataFrame({
            "Medida": ["Variância populacional", "Variância amostral", "Desvio padrão populacional", "Desvio padrão amostral"],
            "Valor": [var_pop, var_sample, std_pop, std_sample],
        })
        st.dataframe(dispersion_table, use_container_width=True, hide_index=True)

        st.subheader("Quartis e percentil")
        quartile_table = pd.DataFrame({
            "Medida": ["Q1 (25%)", "Mediana (50%)", "Q3 (75%)", "P90", "IQR"],
            "Valor": [q1_val, q2_val, q3_val, p90_val, iqr_val],
        })
        st.dataframe(quartile_table, use_container_width=True, hide_index=True)

        st.subheader("Tabela de frequência")
        try:
            bins = min(10, max(2, int(np.sqrt(len(data_clean)))))
            frequency = pd.cut(data_clean, bins=bins, include_lowest=True).value_counts().sort_index()
            frequency_table = pd.DataFrame({"Classe": frequency.index.astype(str), "Frequência": frequency.values})
            frequency_table["Frequência relativa (%)"] = (frequency_table["Frequência"] / len(data_clean) * 100).round(2)
            st.dataframe(frequency_table, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.warning(f"Não foi possível construir a tabela de frequência: {exc}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data_clean, kde=True, ax=axes[0])
        axes[0].set_title(f"Histograma — {var_sel}")
        sns.boxplot(x=data_clean, ax=axes[1])
        axes[1].set_title(f"Boxplot — {var_sel}")
        render_figure(fig)

        st.subheader("💡 Interpretação Automática")
        if std_sample == 0:
            st.info("A variável é constante: não há dispersão observada.")
        elif abs(mean_val - med_val) < 0.05 * std_sample:
            st.info("A distribuição aparenta ser aproximadamente simétrica, pois média e mediana são próximas.")
        elif mean_val > med_val:
            st.info("A distribuição apresenta assimetria positiva (à direita), pois a média é maior que a mediana.")
        else:
            st.info("A distribuição apresenta assimetria negativa (à esquerda), pois a média é menor que a mediana.")
        if cv_val < 15:
            st.info(f"O coeficiente de variação é {cv_val:.2f}%, indicando baixa dispersão relativa.")
        elif cv_val < 30:
            st.info(f"O coeficiente de variação é {cv_val:.2f}%, indicando dispersão relativa moderada.")
        else:
            st.info(f"O coeficiente de variação é {cv_val:.2f}%, indicando dispersão relativa elevada.")

    else:
        if not cat_cols:
            st.warning("O dataset não possui variáveis categóricas detectadas.")
            st.stop()
        cat_var = st.selectbox("Selecione uma variável categórica:", cat_cols, key="mod2_cat")
        counts = df[cat_var].fillna("(ausente)").astype(str).value_counts()
        cat_table = pd.DataFrame({
            "Categoria": counts.index,
            "Frequência": counts.values,
            "Frequência relativa (%)": (counts.values / counts.sum() * 100).round(2),
        })
        st.dataframe(cat_table, use_container_width=True, hide_index=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            counts.head(10).sort_values().plot(kind="barh", ax=ax_bar)
            ax_bar.set_title(f"Top categorias — {cat_var}")
            ax_bar.set_xlabel("Frequência")
            render_figure(fig_bar)

        with chart_col2:
            top_counts = counts.head(6).copy()
            if len(counts) > 6:
                top_counts.loc["Outras"] = counts.iloc[6:].sum()
            percentages = top_counts / top_counts.sum() * 100

            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            wedges, _, _ = ax_pie.pie(
                top_counts.values,
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else "",
                startangle=90,
                pctdistance=0.72,
                textprops={"fontsize": 10},
            )

            small_items = []
            for index, (wedge, pct) in enumerate(zip(wedges, percentages)):
                if pct < 5:
                    angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
                    x = math.cos(angle)
                    y = math.sin(angle)
                    small_items.append((index, wedge, pct, x, y))

            left_items = sorted([item for item in small_items if item[3] < 0], key=lambda item: item[4], reverse=True)
            right_items = sorted([item for item in small_items if item[3] >= 0], key=lambda item: item[4], reverse=True)

            def place_small_labels(items, side):
                if not items:
                    return
                base_positions = np.linspace(0.65, 1.15, len(items))
                for (_, wedge, pct, x, _), y_text in zip(items, base_positions[::-1]):
                    sign = -1 if side == "left" else 1
                    x_text = 1.35 * sign
                    angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
                    ax_pie.annotate(
                        f"{pct:.1f}%",
                        xy=(0.94 * math.cos(angle), 0.94 * math.sin(angle)),
                        xytext=(x_text, y_text),
                        ha="right" if side == "left" else "left",
                        va="center",
                        fontsize=9,
                        arrowprops={
                            "arrowstyle": "-",
                            "connectionstyle": "arc3",
                            "shrinkA": 0,
                            "shrinkB": 0,
                        },
                    )

            place_small_labels(left_items, "left")
            place_small_labels(right_items, "right")

            ax_pie.set_title(f"Distribuição das categorias — {cat_var}", pad=12)
            ax_pie.legend(
                wedges,
                [f"{label} ({pct:.1f}%)" for label, pct in zip(top_counts.index, percentages)],
                title="Categorias",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
            ax_pie.set_aspect("equal")
            fig_pie.subplots_adjust(left=0.02, right=0.72, top=0.88, bottom=0.05)
            render_figure(fig_pie)

        st.caption("No gráfico de pizza, percentuais menores que 5% são exibidos externamente com uma linha-guia; os valores também aparecem na legenda.")

with tab3:
    st.header("🎲 Simulação de Monte Carlo")
    sim_choice = st.radio("Selecione o experimento:", ["(a) Lei dos Grandes Números (LGN)", "(b) Teorema Central do Limite (TCL)"], horizontal=True)
    if "(a)" in sim_choice:
        st.subheader("Lei dos Grandes Números — lançamento de dado")
        n_reps = st.slider("Número de lançamentos:", 10, 5000, 500, step=50)
        seed = st.number_input("Semente aleatória:", min_value=0, value=42, step=1)
        rng = np.random.default_rng(seed)
        rolls = rng.integers(1, 7, size=n_reps)
        cum_means = np.cumsum(rolls) / np.arange(1, n_reps + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, n_reps + 1), cum_means, label="Média acumulada")
        ax.axhline(3.5, linestyle="--", label="Valor esperado teórico = 3,5")
        ax.set_xlabel("Número de ensaios")
        ax.set_ylabel("Média")
        ax.legend()
        render_figure(fig)
        st.info(f"Média final após {n_reps} lançamentos: **{cum_means[-1]:.4f}**. O valor teórico é 3,5.")
    else:
        st.subheader("Teorema Central do Limite")
        var_tcl = st.selectbox("Escolha a variável do dataset:", num_cols, key="tcl_var")
        sample_size = st.slider("Tamanho da amostra (n):", 5, 200, 30)
        num_samples = st.slider("Número de amostras simuladas (k):", 100, 5000, 1000)
        seed = st.number_input("Semente aleatória:", min_value=0, value=42, step=1, key="tcl_seed")
        pop_data = df[var_tcl].dropna().astype(float).to_numpy()
        if len(pop_data) < sample_size:
            st.warning(f"A variável possui apenas {len(pop_data)} valores válidos, insuficiente para n={sample_size}.")
            st.stop()
        rng = np.random.default_rng(seed)
        samples = rng.choice(pop_data, size=(num_samples, sample_size), replace=True)
        sample_means = samples.mean(axis=1)
        population_mean = minhastats.mean(pop_data.tolist())
        population_std = minhastats.std_dev(pop_data.tolist(), ddof=1)
        theoretical_se = population_std / math.sqrt(sample_size) if population_std else 0.0
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(sample_means, kde=True, ax=ax, stat="density")
        ax.set_title(f"Distribuição das médias amostrais (n={sample_size}, k={num_samples})")
        render_figure(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Média da população", f"{population_mean:.3f}")
        c2.metric("Média das médias", f"{sample_means.mean():.3f}")
        c3.metric("Erro-padrão teórico", f"{theoretical_se:.3f}")

with tab4:
    st.header("📐 Ajuste de Distribuições Teóricas")
    var_dist = st.selectbox("Escolha a variável:", num_cols, key="dist_var")
    dist_type = st.selectbox("Distribuição candidata:", ["Normal", "Exponencial"])
    data_dist = df[var_dist].dropna().astype(float).tolist()
    if len(data_dist) < 2:
        st.warning("São necessários pelo menos 2 valores válidos.")
        st.stop()
    m_hat = minhastats.mean(data_dist)
    s_hat = minhastats.std_dev(data_dist, ddof=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data_dist, stat="density", ax=ax, label="Dados reais")
    x_vals = np.linspace(min(data_dist), max(data_dist), 300)
    if dist_type == "Normal":
        if s_hat == 0:
            plt.close(fig)
            st.warning("A variável possui desvio padrão zero; não é possível ajustar uma Normal contínua.")
            st.stop()
        pdf = stats.norm.pdf(x_vals, loc=m_hat, scale=s_hat)
        ax.plot(x_vals, pdf, linewidth=2, label=f"Normal (μ={m_hat:.2f}, σ={s_hat:.2f})")
        skewness = pd.Series(data_dist).skew()
        interpretation = f"O ajuste visual tende a ser mais adequado quando os dados são aproximadamente simétricos. A assimetria amostral foi {skewness:.3f}."
    else:
        if min(data_dist) < 0 or m_hat <= 0:
            plt.close(fig)
            st.error("A distribuição Exponencial exige dados não negativos e média positiva. Escolha outra variável.")
            st.stop()
        scale = m_hat
        pdf = stats.expon.pdf(x_vals, loc=0, scale=scale)
        ax.plot(x_vals, pdf, linewidth=2, label=f"Exponencial (λ={1/scale:.4f})")
        interpretation = f"O ajuste Exponencial é plausível quando os dados são não negativos e apresentam decaimento aproximadamente exponencial. A média estimada foi {m_hat:.3f}."
    ax.set_title(f"Ajuste {dist_type} — {var_dist}")
    ax.legend()
    render_figure(fig)
    st.info(interpretation)

with tab5:
    st.header("📈 Correlação e Regressão Linear Simples")
    if len(num_cols) < 2:
        st.warning("São necessárias pelo menos duas variáveis numéricas.")
        st.stop()
    col_x, col_y = st.columns(2)
    x_var = col_x.selectbox("Variável independente (X):", num_cols, index=0)
    y_var = col_y.selectbox("Variável dependente (Y):", num_cols, index=min(1, len(num_cols) - 1))
    df_reg = df[[x_var, y_var]].dropna()
    x_list = df_reg[x_var].astype(float).tolist()
    y_list = df_reg[y_var].astype(float).tolist()
    if len(x_list) < 2:
        st.warning("Não há pares suficientes para o modelo.")
        st.stop()
    try:
        b0, b1, r2 = minhastats.linear_regression(x_list, y_list)
        r = minhastats.pearson_correlation(x_list, y_list)
    except ValueError as exc:
        st.error(f"Não foi possível ajustar o modelo: {exc}")
        st.stop()
    st.latex(fr"\hat{{Y}} = {b0:.4f} + {b1:.4f} \cdot X")
    m1, m2 = st.columns(2)
    m1.metric("Coeficiente de correlação (r)", f"{r:.4f}")
    m2.metric("Coeficiente de determinação (R²)", f"{r2:.4f}")
    st.subheader("🔮 Predição interativa")
    x_pred = st.number_input(f"Digite um valor para {x_var}:", value=float(minhastats.mean(x_list)))
    y_hat = b0 + b1 * x_pred
    st.success(f"**Valor predito para {y_var}: {y_hat:.2f}**")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(x_list, y_list, alpha=0.3, label="Dados reais")
    x_range = np.linspace(min(x_list), max(x_list), 100)
    ax.plot(x_range, b0 + b1 * x_range, linewidth=2, label="Reta OLS")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.legend()
    render_figure(fig)
    strength = "forte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.3 else "fraca"
    st.info(f"A associação linear observada é **{strength}** (|r|={abs(r):.3f}).")
    st.warning("⚠️ Correlação estatística não implica causalidade clínica.")

with tab6:
    st.header("🔎 Módulo 6 — Três descobertas estatísticas")
    st.markdown("As descobertas abaixo são calculadas automaticamente a partir da variável selecionada e servem de apoio ao relatório final.")
    discovery_var = st.selectbox("Variável numérica:", num_cols, key="discovery_var")
    values = df[discovery_var].dropna().astype(float).tolist()
    mean_d = minhastats.mean(values)
    median_d = minhastats.median(values)
    std_d = minhastats.std_dev(values)
    q1_d = minhastats.quantile(values, 0.25)
    q3_d = minhastats.quantile(values, 0.75)
    iqr_d = q3_d - q1_d
    outlier_count = sum(x < q1_d - 1.5 * iqr_d or x > q3_d + 1.5 * iqr_d for x in values)
    cv_d = minhastats.coefficient_of_variation(values)

    st.subheader("Descoberta 1 — Tendência central")
    if mean_d > median_d:
        st.write(f"A média ({mean_d:.2f}) é maior que a mediana ({median_d:.2f}), sugerindo assimetria positiva.")
    elif mean_d < median_d:
        st.write(f"A média ({mean_d:.2f}) é menor que a mediana ({median_d:.2f}), sugerindo assimetria negativa.")
    else:
        st.write(f"A média e a mediana são praticamente iguais ({mean_d:.2f}), sugerindo distribuição aproximadamente simétrica.")
    st.subheader("Descoberta 2 — Dispersão")
    st.write(f"O desvio padrão amostral é {std_d:.2f} e o coeficiente de variação é {cv_d:.2f}%, indicando a magnitude da variabilidade relativa.")
    st.subheader("Descoberta 3 — Valores extremos")
    st.write(f"A regra do IQR identificou **{outlier_count}** outliers entre {len(values)} observações ({outlier_count / len(values) * 100:.2f}%).")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(values, kde=True, ax=ax)
    ax.axvline(mean_d, linestyle="--", label="Média")
    ax.axvline(median_d, linestyle=":", label="Mediana")
    ax.legend()
    render_figure(fig)
