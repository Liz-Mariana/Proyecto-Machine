"""Análisis de generalización, sobreajuste, estabilidad y pruebas estadísticas."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve, validation_curve

from . import config as C
from .models import MODEL_NAMES
from .plots import TEXT_SECONDARY, save, set_style
from .tuning import cv_splitter


# ---------------------------------------------------------------------------
# Riesgo empírico vs. riesgo de generalización (pérdida 0-1)
# ---------------------------------------------------------------------------
def risk_table(models: dict, results: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Con la pérdida 0-1, el riesgo es la tasa de error = 1 - accuracy.

    - Riesgo empírico R_emp: error sobre los datos con los que se entrenó.
    - Riesgo estimado por CV: error medio en los folds de validación.
    - Riesgo de prueba: estimación insesgada del riesgo real R(f).
    """
    rows = []
    for name in MODEL_NAMES:
        m = models[name]
        cv_acc = cross_val_score(clone(m), X_train, y_train, cv=cv_splitter(), scoring="accuracy",
                                 n_jobs=C.N_JOBS)
        r_emp = 1 - accuracy_score(y_train, m.predict(X_train))
        r_cv = 1 - cv_acc.mean()
        r_test = 1 - accuracy_score(y_test, m.predict(X_test))
        rows.append({"modelo": name, "riesgo_empirico": r_emp, "riesgo_cv": r_cv, "riesgo_prueba": r_test,
                     "brecha_generalizacion": r_test - r_emp})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "riesgo_empirico_vs_real.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Curvas de aprendizaje y de validación (complejidad)
# ---------------------------------------------------------------------------
def learning_curves(models: dict, X_train, y_train) -> pd.DataFrame:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    rows = []
    for ax, name in zip(axes, MODEL_NAMES):
        sizes, tr, va = learning_curve(clone(models[name]), X_train, y_train, cv=cv_splitter(),
                                       scoring=C.SCORING, train_sizes=np.linspace(0.1, 1.0, 8),
                                       n_jobs=C.N_JOBS, shuffle=True, random_state=C.RANDOM_STATE)
        for s, a, b, c, d in zip(sizes, tr.mean(1), tr.std(1), va.mean(1), va.std(1)):
            rows.append({"modelo": name, "n_entrenamiento": s, "f1_train": a, "f1_train_std": b,
                         "f1_validacion": c, "f1_validacion_std": d})
        for mean, std, label, color in ((tr.mean(1), tr.std(1), "Entrenamiento", "#2a78d6"),
                                        (va.mean(1), va.std(1), "Validación cruzada", "#eb6834")):
            ax.plot(sizes, mean, "o-", color=color, label=label, markersize=5)
            ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("Tamaño del conjunto de entrenamiento")
        ax.set_ylim(0.3, 1.03)
        ax.text(0.97, 0.04, f"brecha final (train - CV) = {tr.mean(1)[-1] - va.mean(1)[-1]:.3f}",
                transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_SECONDARY)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(loc="upper right")
    fig.suptitle("Curvas de aprendizaje de los modelos optimizados", fontweight="bold")
    fig.tight_layout()
    save(fig, "13_curvas_aprendizaje.png")
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "curvas_aprendizaje.csv", index=False)
    return out


VALIDATION_PARAMS = {
    "Regresión logística": ("clf__C", np.logspace(-3, 3, 13), "C (inverso de la regularización, escala log)"),
    "Árbol de decisión": ("clf__max_depth", np.arange(1, 21), "max_depth (profundidad máxima)"),
    "Random Forest": ("clf__max_depth", np.array([1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30]),
                      "max_depth (profundidad máxima de cada árbol)"),
}


def validation_curves(models: dict, X_train, y_train) -> pd.DataFrame:
    """Varía un hiperparámetro de complejidad dejando los demás en su valor óptimo."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    rows = []
    for ax, name in zip(axes, MODEL_NAMES):
        param, values, label = VALIDATION_PARAMS[name]
        tr, va = validation_curve(clone(models[name]), X_train, y_train, param_name=param, param_range=values,
                                  cv=cv_splitter(), scoring=C.SCORING, n_jobs=C.N_JOBS)
        for v, a, b in zip(values, tr.mean(1), va.mean(1)):
            rows.append({"modelo": name, "parametro": param.replace("clf__", ""), "valor": v,
                         "f1_train": a, "f1_validacion": b})
        for mean, std, lab, color in ((tr.mean(1), tr.std(1), "Entrenamiento", "#2a78d6"),
                                      (va.mean(1), va.std(1), "Validación cruzada", "#eb6834")):
            ax.plot(values, mean, "o-", color=color, label=lab, markersize=4)
            ax.fill_between(values, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        chosen = models[name].get_params()[param]
        if chosen is not None:
            ax.axvline(chosen, color="#52514e", linestyle=":", linewidth=1.2)
            ax.text(chosen, 0.32, f"elegido = {chosen:.4g} ", fontsize=8, color=TEXT_SECONDARY,
                    ha="right" if name == "Regresión logística" else "left")
        else:
            ax.text(values[-1], 0.32, "elegido = None (sin límite)", fontsize=8, ha="right", color=TEXT_SECONDARY)
        if name == "Regresión logística":
            ax.set_xscale("log")
        else:
            ax.set_xticks(values if len(values) <= 12 else values[::2])
        ax.set_xlabel(label)
        ax.set_title(name)
        ax.set_ylim(0.3, 1.03)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(loc="upper left")
    fig.suptitle("Curvas de validación: complejidad del modelo vs. rendimiento", fontweight="bold")
    fig.tight_layout()
    save(fig, "14_curvas_validacion.png")
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "curvas_validacion.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Estabilidad (CV repetida) y comparación estadística
# ---------------------------------------------------------------------------
def repeated_cv(models: dict, X_train, y_train) -> pd.DataFrame:
    """CV estratificada 5x5: las mismas 25 particiones para los tres modelos (datos pareados)."""
    rcv = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, random_state=C.RANDOM_STATE)
    scores = {name: cross_val_score(clone(models[name]), X_train, y_train, cv=rcv, scoring=C.SCORING,
                                    n_jobs=C.N_JOBS) for name in MODEL_NAMES}
    out = pd.DataFrame(scores)
    out.to_csv(C.TAB_DIR / "cv_repetida_scores.csv", index_label="particion")
    summary = out.agg(["mean", "std", "min", "max"]).T
    summary["coef_variacion_%"] = 100 * summary["std"] / summary["mean"]
    summary.round(4).to_csv(C.TAB_DIR / "cv_repetida_resumen.csv", index_label="modelo")

    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot([out[n] for n in MODEL_NAMES], patch_artist=True, widths=0.5,
                    medianprops={"color": "#0b0b0b"}, showfliers=False)
    rng = np.random.default_rng(C.RANDOM_STATE)
    for k, (patch, name) in enumerate(zip(bp["boxes"], MODEL_NAMES), start=1):
        patch.set_facecolor(C.MODEL_COLORS[name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(C.MODEL_COLORS[name])
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(out)), out[name], s=14, color=C.MODEL_COLORS[name],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(k + 0.3, out[name].mean(), f"media {out[name].mean():.3f}\nde {out[name].std():.3f}",
                fontsize=8, va="center", color=TEXT_SECONDARY)
    ax.set_xticks(range(1, 4), MODEL_NAMES)
    ax.set_xlim(0.5, 3.9)
    ax.set_ylabel("F1 macro por partición")
    ax.set_title("Estabilidad: validación cruzada estratificada repetida 5x5 (25 particiones)")
    ax.grid(axis="x", visible=False)
    save(fig, "15_estabilidad_cv_repetida.png")
    return out


def corrected_t_test(scores: pd.DataFrame, n_train: int) -> pd.DataFrame:
    """t-test pareado corregido de Nadeau y Bengio (2003).

    Las particiones de la CV comparten observaciones, por lo que el t-test clásico
    subestima la varianza. La corrección multiplica la varianza por
    (1/J + n_val/n_train), con J = número de particiones.
    """
    J = len(scores)
    ratio = (1 / C.CV_FOLDS) / (1 - 1 / C.CV_FOLDS)   # n_val / n_train dentro de cada fold
    rows = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = MODEL_NAMES[i], MODEL_NAMES[j]
            d = scores[a] - scores[b]
            var = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / J + ratio) * var)
            p = 2 * stats.t.sf(abs(t), df=J - 1)
            rows.append({"modelo_A": a, "modelo_B": b, "diferencia_media_F1": d.mean(),
                         "t_corregido": t, "p_valor": p, "significativo_5%": p < 0.05})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "prueba_t_corregida_cv.csv", index=False)
    return out


def mcnemar_tests(models: dict, X_test, y_test) -> pd.DataFrame:
    """Prueba de McNemar (exacta, binomial) sobre los aciertos/errores en prueba."""
    correct = {n: (models[n].predict(X_test) == y_test.values) for n in MODEL_NAMES}
    rows = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = MODEL_NAMES[i], MODEL_NAMES[j]
            n01 = int(np.sum(correct[a] & ~correct[b]))   # A acierta, B falla
            n10 = int(np.sum(~correct[a] & correct[b]))   # A falla, B acierta
            p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
            rows.append({"modelo_A": a, "modelo_B": b, "A_acierta_B_falla": n01, "A_falla_B_acierta": n10,
                         "p_valor": p, "significativo_5%": p < 0.05})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "prueba_mcnemar_test.csv", index=False)
    return out


def bootstrap_ci(models: dict, X_test, y_test, n_boot: int = 2000) -> pd.DataFrame:
    """Intervalos de confianza del 95 % (percentil) para accuracy y F1 macro en prueba."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    y = y_test.values
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for name in MODEL_NAMES:
        pred = models[name].predict(X_test)
        accs = np.array([accuracy_score(y[k], pred[k]) for k in idx])
        f1s = np.array([f1_score(y[k], pred[k], average="macro") for k in idx])
        rows.append({"modelo": name,
                     "accuracy": accuracy_score(y, pred), "accuracy_IC95_inf": np.percentile(accs, 2.5),
                     "accuracy_IC95_sup": np.percentile(accs, 97.5),
                     "f1_macro": f1_score(y, pred, average="macro"), "f1_IC95_inf": np.percentile(f1s, 2.5),
                     "f1_IC95_sup": np.percentile(f1s, 97.5)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "intervalos_confianza_test.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Interpretación: importancia por permutación
# ---------------------------------------------------------------------------
def permutation_importances(models: dict, X_test, y_test) -> pd.DataFrame:
    """Caída del F1 macro en prueba al permutar cada variable original (20 repeticiones)."""
    rows = []
    for name in MODEL_NAMES:
        r = permutation_importance(models[name], X_test, y_test, scoring=C.SCORING, n_repeats=20,
                                   random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS)
        for col, m, s in zip(X_test.columns, r.importances_mean, r.importances_std):
            rows.append({"modelo": name, "variable": col, "importancia_media": m, "importancia_std": s})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "importancia_permutacion.csv", index=False)

    set_style()
    order = (out.groupby("variable")["importancia_media"].mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(order))
    h = 0.27
    for k, name in enumerate(MODEL_NAMES):
        d = out[out["modelo"] == name].set_index("variable").loc[order]
        ax.barh(y + (k - 1) * h, d["importancia_media"], h - 0.02, xerr=d["importancia_std"],
                color=C.MODEL_COLORS[name], label=name,
                error_kw={"ecolor": "#52514e", "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(y, order)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_xlabel("Disminución del F1 macro en prueba al permutar la variable")
    ax.set_title("Importancia de las variables por permutación")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "16_importancia_permutacion.png")
    return out
