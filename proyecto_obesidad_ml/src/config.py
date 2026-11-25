"""Configuración central del proyecto: rutas, semillas y definición de variables.

Cambiar un parámetro aquí (p. ej. el tamaño del conjunto de prueba o el número de
folds) se propaga automáticamente a todo el proyecto.
"""
import os
import warnings
from pathlib import Path

# Cuando una categoría muy rara (p. ej. MTRANS = "Bike") no aparece en el fold de
# entrenamiento, el OneHotEncoder la codifica como ceros (handle_unknown="ignore").
# Es el comportamiento deseado; se silencia el aviso para mantener limpia la salida,
# tanto en este proceso como en los procesos paralelos (variable de entorno).
_ENC_WARNING = "ignore::UserWarning:sklearn.preprocessing._encoders"
os.environ["PYTHONWARNINGS"] = _ENC_WARNING
warnings.filterwarnings("ignore", message="Found unknown categories")

# ---------------------------------------------------------------------------
# Rutas (relativas a la raíz del proyecto, funcionan en cualquier computador)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "ObesityDataSet_raw_and_data_sinthetic.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

for _d in (DATA_PROCESSED, MODELS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibilidad y validación
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 % entrenamiento / 20 % prueba
CV_FOLDS = 5              # validación cruzada estratificada de 5 folds
CV_REPEATS = 5            # repeticiones para el análisis de estabilidad (5x5)
SCORING = "f1_macro"      # métrica guía de la búsqueda de hiperparámetros
N_JOBS = -1

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
TARGET = "NObeyesdad"

# Orden natural (ordinal) de las clases: de menor a mayor IMC
CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
CLASS_SHORT = ["Insuf.", "Normal", "Sobrep. I", "Sobrep. II", "Obes. I", "Obes. II", "Obes. III"]

# Variables antropométricas: definen el IMC y, por tanto, la etiqueta
ANTHROPOMETRIC = ["Height", "Weight"]

# Numéricas de hábitos / demografía (escenario principal)
NUMERIC_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Categóricas ordinales (con su orden lógico)
ORDINAL_FEATURES = {
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
}

# Categóricas nominales
NOMINAL_FEATURES = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]

# Descripción de cada variable (para tablas y el informe)
VARIABLE_DESCRIPTIONS = {
    "Gender": "Género",
    "Age": "Edad (años)",
    "Height": "Estatura (m)",
    "Weight": "Peso (kg)",
    "family_history_with_overweight": "Antecedente familiar de sobrepeso",
    "FAVC": "Consumo frecuente de alimentos hipercalóricos",
    "FCVC": "Frecuencia de consumo de verduras (1-3)",
    "NCP": "Número de comidas principales al día (1-4)",
    "CAEC": "Consumo de alimentos entre comidas",
    "SMOKE": "Fuma",
    "CH2O": "Consumo diario de agua (1-3)",
    "SCC": "Monitorea las calorías que consume",
    "FAF": "Frecuencia de actividad física (0-3)",
    "TUE": "Tiempo de uso de dispositivos tecnológicos (0-2)",
    "CALC": "Consumo de alcohol",
    "MTRANS": "Medio de transporte habitual",
    "NObeyesdad": "Nivel de obesidad (variable objetivo)",
}

# ---------------------------------------------------------------------------
# Estilo de gráficas (paleta validada para daltonismo)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "Regresión logística": "#2a78d6",
    "Árbol de decisión": "#eb6834",
    "Random Forest": "#1baf7a",
}
# Rampa secuencial (azul) para las 7 clases ordenadas por nivel de IMC
CLASS_COLORS = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
