# Clasificación multiclase del nivel de obesidad

Proyecto universitario de Machine Learning. Predice la variable `NObeyesdad` (7 niveles de peso y
obesidad) a partir de hábitos alimenticios, actividad física y datos demográficos. Compara tres modelos:
**regresión logística multinomial**, **árbol de decisión** y **Random Forest**, con énfasis en la
**generalización** y el **sobreajuste**.

## Resultados principales (conjunto de prueba, 418 observaciones)

| Modelo | F1 entren. | F1 validación cruzada | F1 prueba | Accuracy prueba | Diagnóstico |
|---|---|---|---|---|---|
| Regresión logística | 0,578 | 0,555 | 0,549 | 0,586 | Subajuste (sesgo alto) |
| Árbol de decisión | 1,000 | 0,749 | 0,724 | 0,730 | Sobreajuste (varianza alta) |
| **Random Forest** | 0,991 | **0,854** | **0,852** | **0,857** | **Modelo seleccionado** |

Decisión clave: `Weight` y `Height` se **excluyen** de los modelos. La etiqueta se construyó con el IMC
(= peso / estatura²) y el 95,1 % de las etiquetas coincide con la regla de la OMS, así que incluirlas
produce fuga de información del objetivo (ver sección 5.4 del informe).

## Estructura

```
proyecto_obesidad_ml/
├── data/
│   ├── raw/ObesityDataSet_raw_and_data_sinthetic.csv   # datos originales (no se modifican)
│   └── processed/train.csv, test.csv                   # partición estratificada 80/20 (sin duplicados)
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb                  # EDA + análisis de Weight/Height
│   └── 02_modelamiento_y_generalizacion.ipynb          # modelos, optimización, evaluación, generalización
├── src/
│   ├── config.py           # rutas, semilla, variables, parámetros de CV
│   ├── data.py             # carga, limpieza, partición
│   ├── preprocessing.py    # ColumnTransformer (escalado + codificación)
│   ├── models.py           # pipelines y espacios de búsqueda de hiperparámetros
│   ├── eda.py              # tablas y gráficas del análisis exploratorio
│   ├── leakage.py          # análisis del IMC / Weight-Height
│   ├── tuning.py           # Grid Search y Random Search
│   ├── evaluation.py       # métricas, matrices de confusión, comparaciones
│   ├── generalization.py   # riesgos, curvas, estabilidad, pruebas estadísticas
│   ├── plots.py            # estilo común de las gráficas
│   └── pipeline.py         # orquesta todas las etapas
├── models/                 # modelos entrenados (.joblib): pipeline completo preprocesamiento + modelo
├── results/
│   ├── figures/            # 16 gráficas (PNG)
│   ├── tables/             # tablas de resultados (CSV)
│   └── resumen_ejecucion.json
├── reports/
│   ├── informe_final.pdf   # informe académico
│   ├── informe_final.docx  # versión editable en Word
│   └── informe_final.md    # fuente del informe
├── main.py                 # ejecuta todo el experimento
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo en VS Code

**Requisitos:** Python 3.11 o superior y VS Code con las extensiones **Python** y **Jupyter** (Microsoft).

1. Descomprime el ZIP y abre la carpeta `proyecto_obesidad_ml` en VS Code (*Archivo → Abrir carpeta*).
2. Abre una terminal (*Terminal → Nueva terminal*) y crea un entorno virtual:

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Si PowerShell bloquea la activación, ejecuta antes:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Selecciona el intérprete: `Ctrl+Shift+P` → *Python: Select Interpreter* → elige el de `.venv`.
4. Ejecuta el experimento completo (≈ 5 minutos):
   ```bash
   python main.py
   ```
   También puedes usar *Ejecutar y depurar* (`F5`) → **"Ejecutar proyecto completo (main.py)"**.
   El script regenera `data/processed/`, `models/` y `results/`.
5. Para los notebooks, abre `notebooks/01_...ipynb` o `02_...ipynb`, elige el kernel `.venv` (arriba a la
   derecha) y pulsa **Run All**. Ya incluyen las salidas de la ejecución, así que se pueden leer sin
   ejecutarlos.

## Usar el modelo guardado

```python
import joblib, pandas as pd
modelo = joblib.load("models/random_forest.joblib")        # incluye el preprocesamiento
nuevos = pd.read_csv("data/processed/test.csv").head()
print(modelo.predict(nuevos))                               # usa solo las columnas que necesita
```

## Decisiones metodológicas (resumen)

| Decisión | Motivo |
|---|---|
| Eliminar 24 duplicados antes de dividir | Evitar que un mismo registro esté en entrenamiento y prueba |
| Excluir `Weight` y `Height` | La etiqueta es una función de ellas (IMC): fuga de información del objetivo |
| Partición estratificada 80/20, semilla 42 | Conservar proporciones de clase; reproducibilidad |
| Preprocesamiento dentro de `Pipeline` | Escalado y codificación se ajustan solo con el fold de entrenamiento |
| `CAEC`, `CALC` codificadas ordinalmente | Tienen orden natural (no < Sometimes < Frequently < Always) |
| Estandarización solo en regresión logística | La penalización depende de la escala; los árboles son invariantes |
| Métrica guía F1 macro | Pondera por igual las 7 clases |
| CV estratificada de 5 folds para la búsqueda | El conjunto de prueba no interviene en la selección |
| CV repetida 5×5 + *t*-test corregido | Medir estabilidad y comparar modelos con folds solapados |

## Reproducibilidad

Todas las fuentes de azar usan `random_state = 42`. Resultados obtenidos con Python 3.13, scikit-learn
1.9.1, pandas 3.0.5 y numpy 2.5.3 (ver `results/resumen_ejecucion.json`). Con otras versiones de las
librerías pueden aparecer diferencias mínimas en el tercer decimal.

## Datos

Palechor, F. M., y De la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating
habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief, 25*, 104344.
(UCI Machine Learning Repository.)
