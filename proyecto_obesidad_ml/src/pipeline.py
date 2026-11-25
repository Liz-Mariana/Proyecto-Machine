"""Orquestación del experimento completo (lo usan main.py y los notebooks)."""
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn

from . import config as C
from . import data, eda, evaluation, generalization, leakage, tuning

warnings.filterwarnings("ignore", category=FutureWarning)


def step_data():
    df_raw = data.load_raw()
    df = data.clean(df_raw)
    train_df, test_df = data.split(df)
    return df_raw, df, train_df, test_df


def step_modeling(train_df, test_df, verbose=True):
    X_train, y_train = data.get_xy(train_df)
    X_test, y_test = data.get_xy(test_df)
    baseline = tuning.baseline_cv(X_train, y_train)
    results = tuning.run_searches(X_train, y_train, verbose=verbose)
    summary = tuning.search_summary(results, baseline)
    models = tuning.load_best_models()
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "baseline": baseline, "results": results, "summary": summary, "models": models}


def step_evaluation(m):
    args = (m["X_train"], m["y_train"], m["X_test"], m["y_test"])
    comp = evaluation.comparison_table(m["models"], m["results"], *args)
    ba = evaluation.before_after_table(m["models"], m["results"], m["baseline"], *args)
    per_class = evaluation.per_class_reports(m["models"], m["X_test"], m["y_test"])
    evaluation.plot_confusion_matrices(m["models"], m["X_test"], m["y_test"])
    evaluation.plot_train_cv_test(comp)
    evaluation.plot_before_after(ba)
    return {"comparacion": comp, "antes_despues": ba, "por_clase": per_class}


def step_generalization(m):
    models = m["models"]
    risks = generalization.risk_table(models, m["results"], m["X_train"], m["y_train"], m["X_test"], m["y_test"])
    lc = generalization.learning_curves(models, m["X_train"], m["y_train"])
    vc = generalization.validation_curves(models, m["X_train"], m["y_train"])
    rcv = generalization.repeated_cv(models, m["X_train"], m["y_train"])
    ttest = generalization.corrected_t_test(rcv, len(m["X_train"]))
    mcn = generalization.mcnemar_tests(models, m["X_test"], m["y_test"])
    ci = generalization.bootstrap_ci(models, m["X_test"], m["y_test"])
    imp = generalization.permutation_importances(models, m["X_test"], m["y_test"])
    return {"riesgos": risks, "curvas_aprendizaje": lc, "curvas_validacion": vc, "cv_repetida": rcv,
            "t_test": ttest, "mcnemar": mcn, "ic_bootstrap": ci, "importancia": imp}


def run_all():
    t0 = time.perf_counter()
    print("[1/5] Carga y limpieza de datos")
    df_raw, df, train_df, test_df = step_data()
    print(f"      {df_raw.shape[0]} filas originales -> {df.shape[0]} sin duplicados "
          f"({len(train_df)} entrenamiento / {len(test_df)} prueba)")

    print("[2/5] Análisis exploratorio")
    info = eda.run_eda(df_raw, df)

    print("[3/5] Análisis de Weight y Height (fuga de información)")
    leak = leakage.run_leakage_analysis(df, train_df)
    print(f"      Coincidencia etiqueta vs. regla OMS del IMC: {100 * leak['coincidencia_regla_OMS']:.1f} %")

    print("[4/5] Optimización de hiperparámetros (Grid Search y Random Search)")
    m = step_modeling(train_df, test_df)
    ev = step_evaluation(m)
    print(ev["comparacion"][["modelo", "f1_train", "f1_cv_media", "f1_test", "accuracy_test"]].to_string(index=False))

    print("[5/5] Generalización, estabilidad y pruebas estadísticas")
    gen = step_generalization(m)

    summary = {
        "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M"),
        "duracion_min": round((time.perf_counter() - t0) / 60, 2),
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
        "semilla": C.RANDOM_STATE,
        "datos": info | {"n_train": len(train_df), "n_test": len(test_df)},
        "coincidencia_regla_OMS": round(leak["coincidencia_regla_OMS"], 4),
        "modelos": ev["comparacion"].to_dict(orient="records"),
    }
    with open(C.ROOT / "results" / "resumen_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Listo en {summary['duracion_min']} min. Resultados en results/, modelos en models/.")
    return {"info": info, "leak": leak, "modeling": m, "evaluation": ev, "generalization": gen}
