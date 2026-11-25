"""Optimización de hiperparámetros con Grid Search y Random Search.

Ambas búsquedas usan validación cruzada estratificada sobre el conjunto de
ENTRENAMIENTO. El conjunto de prueba no participa en ninguna decisión.
"""
import time
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate

from . import config as C
from .models import GRID_SPACES, MODEL_FILES, MODEL_NAMES, N_ITER, RANDOM_SPACES, build_pipeline


def cv_splitter():
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def _clean_params(params: dict) -> str:
    out = {}
    for k, v in params.items():
        k = k.replace("clf__", "")
        if isinstance(v, float):
            v = round(v, 4)
        out[k] = v
    return str(out)


def baseline_cv(X, y) -> dict:
    """Validación cruzada de los modelos con hiperparámetros por defecto."""
    res = {}
    for name in MODEL_NAMES:
        cv = cross_validate(build_pipeline(name), X, y, cv=cv_splitter(), scoring=C.SCORING,
                            return_train_score=True, n_jobs=C.N_JOBS)
        res[name] = {
            "f1_train_cv": cv["train_score"].mean(),
            "f1_cv_media": cv["test_score"].mean(),
            "f1_cv_std": cv["test_score"].std(),
        }
    return res


def run_searches(X, y, verbose: bool = True) -> dict:
    """Ejecuta Grid Search y Random Search para los tres modelos.

    Devuelve un diccionario con, para cada modelo, ambos objetos de búsqueda y el
    mejor pipeline seleccionado (el de mayor F1 macro en validación cruzada).
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out = {}
    for name in MODEL_NAMES:
        searches = {}
        for method in ("Grid Search", "Random Search"):
            if method == "Grid Search":
                search = GridSearchCV(build_pipeline(name), GRID_SPACES[name], scoring=C.SCORING,
                                      cv=cv_splitter(), n_jobs=C.N_JOBS, return_train_score=True)
            else:
                search = RandomizedSearchCV(build_pipeline(name), RANDOM_SPACES[name], n_iter=N_ITER[name],
                                            scoring=C.SCORING, cv=cv_splitter(), n_jobs=C.N_JOBS,
                                            return_train_score=True, random_state=C.RANDOM_STATE)
            t0 = time.perf_counter()
            search.fit(X, y)
            search.elapsed_ = time.perf_counter() - t0
            searches[method] = search
            if verbose:
                print(f"  {name:<20} {method:<14} F1 CV = {search.best_score_:.4f}  "
                      f"({len(search.cv_results_['params'])} combinaciones, {search.elapsed_:.1f} s)")
            # Se guardan las 10 mejores combinaciones de cada búsqueda
            res = pd.DataFrame(search.cv_results_)
            cols = [c for c in res.columns if c.startswith("param_")] + [
                "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
            res[cols].sort_values("rank_test_score").head(10).to_csv(
                C.TAB_DIR / f"busqueda_{MODEL_FILES[name]}_{'grid' if method == 'Grid Search' else 'random'}.csv",
                index=False)
        # Criterio de selección: mayor F1 macro en CV (empate -> Grid Search)
        best_method = max(searches, key=lambda m: (round(searches[m].best_score_, 6), m == "Grid Search"))
        best = searches[best_method]
        joblib.dump(best.best_estimator_, C.MODELS_DIR / f"{MODEL_FILES[name]}.joblib")
        out[name] = {"searches": searches, "best_method": best_method, "best": best}
    return out


def search_summary(results: dict, baseline: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({"modelo": name, "metodo": "Por defecto (sin optimizar)", "combinaciones": 1,
                     "tiempo_s": None, "f1_train_cv": baseline[name]["f1_train_cv"],
                     "f1_cv_media": baseline[name]["f1_cv_media"], "f1_cv_std": baseline[name]["f1_cv_std"],
                     "mejores_hiperparametros": "por defecto", "seleccionado": False})
        for method, s in r["searches"].items():
            i = s.best_index_
            rows.append({"modelo": name, "metodo": method, "combinaciones": len(s.cv_results_["params"]),
                         "tiempo_s": round(s.elapsed_, 1),
                         "f1_train_cv": s.cv_results_["mean_train_score"][i],
                         "f1_cv_media": s.best_score_, "f1_cv_std": s.cv_results_["std_test_score"][i],
                         "mejores_hiperparametros": _clean_params(s.best_params_),
                         "seleccionado": method == r["best_method"]})
    out = pd.DataFrame(rows)
    for c in ("f1_train_cv", "f1_cv_media", "f1_cv_std"):
        out[c] = out[c].astype(float).round(4)
    out.to_csv(C.TAB_DIR / "optimizacion_resumen.csv", index=False)
    return out


def load_best_models() -> dict:
    """Carga los modelos finales guardados en models/."""
    return {name: joblib.load(C.MODELS_DIR / f"{MODEL_FILES[name]}.joblib") for name in MODEL_NAMES}
