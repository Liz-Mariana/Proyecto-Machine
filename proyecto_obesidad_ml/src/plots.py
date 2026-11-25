"""Utilidades comunes de gráficas."""
import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en scripts y en notebooks
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Mapa divergente (naranja - gris neutro - azul) para correlaciones
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c2491c", "#eb6834", "#f2f1ee", "#2a78d6", "#184f95"])
# Mapa secuencial de un solo tono (azul) para matrices de confusión
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f7fafe", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": "#b9b8b3",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str):
    """Guarda la figura en results/figures y la devuelve (para mostrarla en notebooks)."""
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
