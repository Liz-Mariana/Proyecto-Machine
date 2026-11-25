"""Evaluación de los modelos: métricas, tablas comparativas y matrices de confusión."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from . import config as C
from .models import MODEL_FILES, MODEL_NAMES, build_pipeline
from .plots import SEQUENTIAL, TEXT_SECONDARY, save, set_style


def metrics(y_true, y_pred) -> dict:
    """Métricas multiclase con promedio macro (todas las clases pesan igual)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def comparison_table(models: dict, results: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Tabla principal: entrenamiento vs. validación cruzada vs. prueba."""
    rows = []
    for name in MODEL_NAMES:
        m = models[name]
        best = results[name]["best"]
        i = best.best_index_
        tr = metrics(y_train, m.predict(X_train))
        te = metrics(y_test, m.predict(X_test))
        rows.append({
            "modelo": name,
            "busqueda_seleccionada": results[name]["best_method"],
            "accuracy_train": tr["accuracy"], "accuracy_test": te["accuracy"],
            "precision_test": te["precision_macro"], "recall_test": te["recall_macro"],
            "f1_train": tr["f1_macro"],
            "f1_cv_media": best.best_score_, "f1_cv_std": best.cv_results_["std_test_score"][i],
            "f1_test": te["f1_macro"],
            "brecha_train_test_f1": tr["f1_macro"] - te["f1_macro"],
            "brecha_train_cv_f1": tr["f1_macro"] - best.best_score_,
        })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(C.TAB_DIR / "comparacion_modelos.csv", index=False)
    return out


def before_after_table(models: dict, results: dict, baseline: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Compara cada modelo con hiperparámetros por defecto vs. optimizados."""
    rows = []
    for name in MODEL_NAMES:
        default = build_pipeline(name).fit(X_train, y_train)
        for label, m, cv_score in (("Por defecto", default, baseline[name]["f1_cv_media"]),
                                   ("Optimizado", models[name], results[name]["best"].best_score_)):
            rows.append({
                "modelo": name, "version": label,
                "f1_train": f1_score(y_train, m.predict(X_train), average="macro"),
                "f1_cv": cv_score,
                "f1_test": f1_score(y_test, m.predict(X_test), average="macro"),
                "accuracy_test": accuracy_score(y_test, m.predict(X_test)),
            })
    out = pd.DataFrame(rows)
    out["brecha_train_test"] = out["f1_train"] - out["f1_test"]
    out = out.round(4)
    out.to_csv(C.TAB_DIR / "antes_despues_optimizacion.csv", index=False)
    return out


def per_class_reports(models: dict, X_test, y_test) -> pd.DataFrame:
    frames = []
    for name in MODEL_NAMES:
        rep = classification_report(y_test, models[name].predict(X_test), labels=C.CLASS_ORDER,
                                    output_dict=True, zero_division=0)
        df = pd.DataFrame(rep).T.loc[C.CLASS_ORDER, ["precision", "recall", "f1-score", "support"]]
        df.insert(0, "modelo", name)
        frames.append(df)
    out = pd.concat(frames).rename_axis("clase").reset_index().round(4)
    out.to_csv(C.TAB_DIR / "metricas_por_clase_test.csv", index=False)
    return out


def plot_confusion_matrices(models: dict, X_test, y_test):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, name in zip(axes, MODEL_NAMES):
        cm = confusion_matrix(y_test, models[name].predict(X_test), labels=C.CLASS_ORDER)
        pd.DataFrame(cm, index=C.CLASS_ORDER, columns=C.CLASS_ORDER).to_csv(
            C.TAB_DIR / f"matriz_confusion_{MODEL_FILES[name]}.csv")
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{cm[i, j]}\n{100 * cm_norm[i, j]:.0f}%", ha="center", va="center",
                            fontsize=7, color="white" if cm_norm[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(7), C.CLASS_SHORT, rotation=45, ha="right")
        ax.set_yticks(range(7), C.CLASS_SHORT)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.grid(False)
        acc = accuracy_score(y_test, models[name].predict(X_test))
        ax.set_title(f"{name}  (accuracy prueba = {acc:.3f})")
    fig.suptitle("Matrices de confusión en el conjunto de prueba (conteo y % por fila)", fontweight="bold", y=1.03)
    fig.tight_layout()
    return save(fig, "10_matrices_confusion.png")


def plot_train_cv_test(comp: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(comp))
    w = 0.26
    series = [("Entrenamiento", "f1_train", "#2a78d6"),
              ("Validación cruzada", "f1_cv_media", "#eb6834"),
              ("Prueba", "f1_test", "#1baf7a")]
    for k, (label, col, color) in enumerate(series):
        vals = comp[col].values
        offs = comp["f1_cv_std"].values if col == "f1_cv_media" else np.zeros(len(vals))
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=color, label=label)
        for b, v, o in zip(bars, vals, offs):
            ax.text(b.get_x() + b.get_width() / 2, v + o + 0.01, f"{v:.3f}", ha="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
    ax.errorbar(x, comp["f1_cv_media"], yerr=comp["f1_cv_std"], fmt="none", ecolor="#0b0b0b",
                capsize=3, elinewidth=1)
    ax.set_xticks(x, comp["modelo"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 macro")
    ax.set_title("Rendimiento en entrenamiento, validación cruzada y prueba (modelos optimizados)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    return save(fig, "11_train_cv_test.png")


def plot_before_after(ba: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        d = ba[ba["modelo"] == name].set_index("version")
        x = np.arange(3)
        for k, (ver, color) in enumerate((("Por defecto", "#b9b8b3"), ("Optimizado", C.MODEL_COLORS[name]))):
            vals = d.loc[ver, ["f1_train", "f1_cv", "f1_test"]].values.astype(float)
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=color, label=ver)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7,
                        color=TEXT_SECONDARY)
        ax.set_xticks(x, ["Entrenamiento", "Valid. cruzada", "Prueba"])
        ax.set_title(name)
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("F1 macro")
    fig.suptitle("Antes y después de la optimización de hiperparámetros", fontweight="bold")
    fig.tight_layout()
    return save(fig, "12_antes_despues_optimizacion.png")
