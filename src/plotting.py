"""Shared chart style. Colours come from the Okabe-Ito palette, which is colour-blind safe."""
import matplotlib.pyplot as plt

from src.config import IMAGES_DIR

SENTIMENT_COLORS = {"Positive": "#009E73", "Neutral": "#999999", "Negative": "#D55E00"}
PRIMARY_COLOR = "#0072B2"
ACCENT_COLOR = "#E69F00"
CATEGORICAL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def apply_style():
    """Apply the project chart style to matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL_COLORS),
        "legend.frameon": False,
    })


def save_figure(file_name, fig=None):
    """Save the current (or given) figure to ``images/``, creating the folder if needed."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(IMAGES_DIR / file_name, bbox_inches="tight")
