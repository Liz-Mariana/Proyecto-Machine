"""Análisis del posible problema de Weight y Height.

La variable objetivo se construyó a partir del IMC = peso / estatura². Si el modelo
recibe peso y estatura, puede "reconstruir" la etiqueta en lugar de aprender la
relación entre hábitos y obesidad. Este módulo cuantifica ese efecto.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data import add_bmi, get_xy
from .models import MODEL_NAMES, build_pipeline
from .plots import TEXT_SECONDARY, save, set_style

# Cortes de la OMS usados en el artículo original del dataset
WHO_CUTS = [18.5, 25.0, 30.0, 35.0, 40.0]


def who_category(bmi: pd.Series) -> pd.Series:
    """Categoría OMS (6 niveles; sobrepeso I y II quedan unidos)."""
    labels = ["Insufficient_Weight", "Normal_Weight", "Overweight",
              "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    return pd.cut(bmi, [-np.inf] + WHO_CUTS + [np.inf], labels=labels, right=False).astype(str)


def bmi_table(df: pd.DataFrame) -> pd.DataFrame:
    d = add_bmi(df)
    t = d.groupby(C.TARGET)["BMI"].agg(["count", "min", "mean", "max", "std"]).reindex(C.CLASS_ORDER).round(2)
    t.to_csv(C.TAB_DIR / "imc_por_clase.csv")
    return t


def who_rule_agreement(df: pd.DataFrame) -> float:
    """% de filas cuya etiqueta coincide con la categoría OMS calculada a partir del IMC."""
    d = add_bmi(df)
    label = d[C.TARGET].replace({"Overweight_Level_I": "Overweight", "Overweight_Level_II": "Overweight"})
    return float((who_category(d["BMI"]) == label).mean())


def scenario_comparison(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compara, con validación cruzada sobre ENTRENAMIENTO, los modelos por defecto
    con y sin Weight/Height, y un árbol que usa únicamente el IMC.
    El conjunto de prueba no se usa aquí."""
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for include in (False, True):
        X, y = get_xy(train_df, include_anthropometric=include)
        for name in MODEL_NAMES:
            s = cross_val_score(build_pipeline(name, include), X, y, cv=cv, scoring=C.SCORING, n_jobs=C.N_JOBS)
            rows.append({"escenario": "Con Weight y Height" if include else "Sin Weight y Height (principal)",
                         "modelo": name, "f1_macro_cv_media": s.mean(), "f1_macro_cv_std": s.std()})
    d = add_bmi(train_df)
    tree_bmi = DecisionTreeClassifier(max_depth=4, random_state=C.RANDOM_STATE)
    s = cross_val_score(tree_bmi, d[["BMI"]], d[C.TARGET], cv=cv, scoring=C.SCORING)
    rows.append({"escenario": "Solo IMC (1 variable)", "modelo": "Árbol de decisión (profundidad 4)",
                 "f1_macro_cv_media": s.mean(), "f1_macro_cv_std": s.std()})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "escenarios_peso_altura.csv", index=False)
    return out


def plot_bmi_by_class(df: pd.DataFrame):
    set_style()
    d = add_bmi(df)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [d.loc[d[C.TARGET] == c, "BMI"] for c in C.CLASS_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "#0b0b0b"},
                    flierprops={"markersize": 3, "markerfacecolor": "#52514e", "markeredgecolor": "none"})
    for patch, color in zip(bp["boxes"], C.CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor("#52514e")
    for cut in WHO_CUTS:
        ax.axhline(cut, color="#eb6834", linestyle="--", linewidth=1.2)
        ax.text(7.55, cut, f"{cut}", va="center", fontsize=8, color="#c2491c")
    ax.set_xticks(range(1, 8), C.CLASS_SHORT)
    ax.set_ylabel("IMC = peso / estatura² (kg/m²)")
    ax.set_title("IMC por clase: las etiquetas siguen los cortes de la OMS (líneas naranjas)")
    return save(fig, "07_imc_por_clase.png")


def plot_weight_height(df: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, color, short in zip(C.CLASS_ORDER, C.CLASS_COLORS, C.CLASS_SHORT):
        s = df[df[C.TARGET] == c]
        ax.scatter(s["Height"], s["Weight"], s=12, color=color, label=short, alpha=0.9,
                   edgecolor="white", linewidth=0.3)
    h = np.linspace(df["Height"].min(), df["Height"].max(), 50)
    for cut in WHO_CUTS:
        ax.plot(h, cut * h ** 2, color="#eb6834", linewidth=1, linestyle="--")
        ax.text(h[-1] + 0.005, cut * h[-1] ** 2, f"IMC {cut}", fontsize=7, color="#c2491c", va="center")
    ax.set_xlabel("Estatura (m)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Peso vs. estatura: las clases quedan separadas por curvas de IMC constante")
    ax.legend(title="Clase", fontsize=8, title_fontsize=8, loc="upper left")
    return save(fig, "08_peso_estatura_clases.png")


def plot_scenarios(table: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{m}\n{e.replace(' (principal)', '')}" for e, m in zip(table["escenario"], table["modelo"])]
    colors = ["#2a78d6" if "Sin" in e else ("#eb6834" if "Con" in e else "#52514e") for e in table["escenario"]]
    y = np.arange(len(table))
    ax.barh(y, table["f1_macro_cv_media"], xerr=table["f1_macro_cv_std"], color=colors, height=0.65,
            error_kw={"ecolor": "#52514e", "capsize": 3, "elinewidth": 1})
    for yi, v, sd in zip(y, table["f1_macro_cv_media"], table["f1_macro_cv_std"]):
        ax.text(v + sd + 0.015, yi, f"{v:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 macro (validación cruzada 5 folds, entrenamiento)")
    ax.set_title("Efecto de incluir Weight y Height (modelos con hiperparámetros por defecto)")
    ax.grid(axis="y", visible=False)
    return save(fig, "09_escenarios_peso_estatura.png")


def run_leakage_analysis(df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    bmi_table(df)
    agreement = who_rule_agreement(df)
    table = scenario_comparison(train_df)
    plot_bmi_by_class(df)
    plot_weight_height(df)
    plot_scenarios(table)
    return {"coincidencia_regla_OMS": agreement, "tabla": table}
