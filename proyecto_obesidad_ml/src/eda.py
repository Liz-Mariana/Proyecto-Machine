"""Análisis exploratorio de datos (EDA): tablas y gráficas."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import config as C
from .plots import DIVERGING, TEXT_SECONDARY, save, set_style

NUM_ALL = C.NUMERIC_FEATURES[:1] + C.ANTHROPOMETRIC + C.NUMERIC_FEATURES[1:]
CAT_ALL = C.NOMINAL_FEATURES[:3] + ["CAEC"] + C.NOMINAL_FEATURES[3:5] + ["CALC", "MTRANS"]


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def overview(df_raw: pd.DataFrame) -> dict:
    """Dimensiones, faltantes y duplicados del dataset original."""
    return {
        "filas": int(df_raw.shape[0]),
        "columnas": int(df_raw.shape[1]),
        "valores_faltantes": int(df_raw.isna().sum().sum()),
        "filas_duplicadas": int(df_raw.duplicated().sum()),
        "filas_tras_limpieza": int(df_raw.drop_duplicates().shape[0]),
    }


def variable_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == C.TARGET:
            kind = "Categórica (objetivo, ordinal)"
        elif col in C.ORDINAL_FEATURES:
            kind = "Categórica ordinal"
        elif df[col].dtype.kind in "if":
            kind = "Numérica"
        else:
            kind = "Categórica nominal" + (" (binaria)" if df[col].nunique() == 2 else "")
        rows.append(
            {
                "variable": col,
                "descripcion": C.VARIABLE_DESCRIPTIONS.get(col, ""),
                "tipo_pandas": str(df[col].dtype),
                "tipo_estadistico": kind,
                "valores_unicos": df[col].nunique(),
                "faltantes": int(df[col].isna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(C.TAB_DIR / "eda_variables.csv", index=False)
    return out


def numeric_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    desc = df[NUM_ALL].describe().T
    desc["asimetria"] = df[NUM_ALL].skew()
    desc["curtosis"] = df[NUM_ALL].kurt()
    desc = desc.round(3)
    desc.to_csv(C.TAB_DIR / "eda_descriptivos_numericos.csv")
    return desc


def categorical_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CAT_ALL + [C.TARGET]:
        vc = df[col].value_counts()
        for level, n in vc.items():
            rows.append({"variable": col, "nivel": level, "n": int(n), "porcentaje": round(100 * n / len(df), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(C.TAB_DIR / "eda_frecuencias_categoricas.csv", index=False)
    return out


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df[C.TARGET].value_counts().reindex(C.CLASS_ORDER)
    out = pd.DataFrame({"n": vc, "porcentaje": (100 * vc / vc.sum()).round(2)})
    out.to_csv(C.TAB_DIR / "eda_distribucion_objetivo.csv")
    return out


def _cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def target_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Asociación de cada predictor con la variable objetivo.

    - Numéricas: prueba de Kruskal-Wallis (no paramétrica, no supone normalidad) y
      tamaño de efecto épsilon² = H / (n - 1).
    - Categóricas: prueba chi-cuadrado de independencia y V de Cramér.
    """
    rows = []
    n = len(df)
    for col in NUM_ALL:
        groups = [g[col].values for _, g in df.groupby(C.TARGET)]
        h, p = stats.kruskal(*groups)
        rows.append({"variable": col, "tipo": "Numérica", "prueba": "Kruskal-Wallis",
                     "estadistico": h, "p_valor": p, "tamano_efecto": h / (n - 1),
                     "medida_efecto": "épsilon²"})
    for col in CAT_ALL:
        table = pd.crosstab(df[col], df[C.TARGET])
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        rows.append({"variable": col, "tipo": "Categórica", "prueba": "Chi-cuadrado",
                     "estadistico": chi2, "p_valor": p, "tamano_efecto": _cramers_v(table),
                     "medida_efecto": "V de Cramér",
                     "pct_celdas_esperado_menor_5": round(100 * (expected < 5).mean(), 1)})
    out = pd.DataFrame(rows).sort_values("tamano_efecto", ascending=False).reset_index(drop=True)
    out.to_csv(C.TAB_DIR / "eda_asociacion_objetivo.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def plot_target(df: pd.DataFrame):
    set_style()
    vc = df[C.TARGET].value_counts().reindex(C.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(C.CLASS_SHORT, vc.values, color="#2a78d6", width=0.65)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}\n({100 * v / vc.sum():.1f} %)",
                ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel("Número de observaciones")
    ax.set_title("Distribución de la variable objetivo (NObeyesdad)")
    ax.set_ylim(0, vc.max() * 1.22)
    ax.grid(axis="x", visible=False)
    return save(fig, "01_distribucion_objetivo.png")


def plot_numeric_histograms(df: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    for ax, col in zip(axes.ravel(), NUM_ALL):
        ax.hist(df[col], bins=30, color="#2a78d6", edgecolor="white", linewidth=0.6)
        ax.set_title(f"{col}", fontsize=10)
        ax.set_xlabel(C.VARIABLE_DESCRIPTIONS[col], fontsize=8)
    fig.suptitle("Distribución de las variables numéricas", fontweight="bold")
    fig.tight_layout()
    return save(fig, "02_histogramas_numericas.png")


def plot_numeric_by_class(df: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUM_ALL):
        data = [df.loc[df[C.TARGET] == c, col] for c in C.CLASS_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.5})
        for patch, color in zip(bp["boxes"], C.CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("#52514e")
        ax.set_xticks(range(1, 8), C.CLASS_SHORT, rotation=60, fontsize=8)
        ax.set_title(col, fontsize=10)
    fig.suptitle("Variables numéricas según el nivel de obesidad (clases ordenadas por IMC)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "03_boxplots_por_clase.png")


def plot_correlation(df: pd.DataFrame):
    set_style()
    tmp = df.copy()
    # Se incluye la clase codificada ordinalmente (0..6) para ver su correlación de Spearman
    tmp["Clase (ordinal)"] = tmp[C.TARGET].map({c: i for i, c in enumerate(C.CLASS_ORDER)})
    for col, cats in C.ORDINAL_FEATURES.items():
        tmp[col] = tmp[col].map({c: i for i, c in enumerate(cats)})
    cols = NUM_ALL + list(C.ORDINAL_FEATURES) + ["Clase (ordinal)"]
    corr = tmp[cols].corr(method="spearman")
    corr.round(3).to_csv(C.TAB_DIR / "eda_correlacion_spearman.csv")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ de Spearman")
    ax.set_title("Correlación de Spearman (numéricas, ordinales y clase)")
    return save(fig, "04_correlacion_spearman.png")


def plot_associations(assoc: pd.DataFrame):
    set_style()
    a = assoc.sort_values("tamano_efecto")
    colors = ["#eb6834" if v in C.ANTHROPOMETRIC else "#2a78d6" for v in a["variable"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(a["variable"] + "  (" + a["medida_efecto"] + ")", a["tamano_efecto"], color=colors, height=0.65)
    for y, v in enumerate(a["tamano_efecto"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Tamaño del efecto (asociación con la clase)")
    ax.set_title("Fuerza de asociación de cada predictor con NObeyesdad\n(naranja: variables antropométricas)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, a["tamano_efecto"].max() * 1.12)
    return save(fig, "05_asociacion_objetivo.png")


def plot_categorical_by_class(df: pd.DataFrame):
    set_style()
    cols = ["family_history_with_overweight", "FAVC", "CAEC", "MTRANS", "Gender", "CALC"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        ct = pd.crosstab(df[C.TARGET], df[col], normalize="index").reindex(C.CLASS_ORDER) * 100
        if col in C.ORDINAL_FEATURES:
            ct = ct[[c for c in C.ORDINAL_FEATURES[col] if c in ct.columns]]
        else:
            ct = ct[ct.sum().sort_values(ascending=False).index]
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
        left = np.zeros(len(ct))
        for k, level in enumerate(ct.columns):
            ax.barh(C.CLASS_SHORT, ct[level], left=left, color=palette[k % len(palette)],
                    edgecolor="white", linewidth=1, label=level, height=0.7)
            left += ct[level].values
        ax.set_title(col, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% dentro de la clase")
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3)
    fig.suptitle("Composición de las variables categóricas por nivel de obesidad", fontweight="bold")
    fig.tight_layout()
    return save(fig, "06_categoricas_por_clase.png")


def run_eda(df_raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Ejecuta todo el EDA. ``df_raw`` = original; ``df`` = sin duplicados."""
    info = overview(df_raw)
    variable_table(df)
    numeric_descriptives(df)
    categorical_frequencies(df)
    target_distribution(df)
    assoc = target_associations(df)
    plot_target(df)
    plot_numeric_histograms(df)
    plot_numeric_by_class(df)
    plot_correlation(df)
    plot_associations(assoc)
    plot_categorical_by_class(df)
    return info
