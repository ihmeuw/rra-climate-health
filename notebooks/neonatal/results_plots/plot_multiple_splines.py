import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rra_climate_health.data import DEFAULT_ROOT


SPLINE_ROOT = Path(DEFAULT_ROOT) / "neonatal_mortality" / "models"

PLOT_SPLINES = [
    ("2026_03_29.08/spline_effect_consumption_pd.parquet", "Model v29.08"),
    (
        "archive/run_2026_03_25/2026_03_26.85/spline_effect_consumption_pd.parquet",
        "Model v26.85",
    ),
]

VAR_DICT = {
    "consumption_pd": "Daily Consumption per capita",
    "days_over_30C_prev_0_mo": "Days over 30°C during birth month",
    "days_over_28C_prev_0_mo": "Days over 28°C during birth month",
    "days_over_28C_prev_3_mo_avg": "Average number of days over 28°C for 3 months prior to birth month",
}

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#d35400"]
CI_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#8e44ad", "#d35400"]


def plot_multiple_splines(
    spline_paths: list[tuple[str, str]],
    var_name: str,
    title: str | None = None,
    filepath: str | None = None,
):
    """
    Plots multiple SCAM spline effects on the same axes for comparison.

    Parameters:
    - spline_paths: list of (parquet_path, label) tuples. Each parquet file
      should have columns ['value', 'effect', 'se'].
    - var_name: Variable name (used for axis label lookup in VAR_DICT).
    - title: Optional plot title override.
    - filepath: If provided, save the figure to this path.
    """
    var_name_plt = VAR_DICT.get(var_name, var_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (path, label) in enumerate(spline_paths):
        effect_df = pd.read_parquet(SPLINE_ROOT / path)
        color = COLORS[i % len(COLORS)]
        ci_color = CI_COLORS[i % len(CI_COLORS)]
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        # Plot the Partial Effect (Spline)
        ax.plot(
            effect_df["value"],
            effect_df["effect"],
            color=color,
            lw=2.5,
            linestyle=ls,
            label=label,
        )

        # Add 95% Confidence Interval (1.96 * SE)
        lower_ci = effect_df["effect"] - (1.96 * effect_df["se"])
        upper_ci = effect_df["effect"] + (1.96 * effect_df["se"])

        ax.fill_between(
            effect_df["value"],
            lower_ci,
            upper_ci,
            color=ci_color,
            alpha=0.1,
        )

    # Aesthetics
    ax.set_xlabel(f"{var_name_plt}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Partial Effect (Log-Odds)", fontsize=12, fontweight="bold")

    full_title = title if title else f"SCAM Spline Effect: {var_name_plt}"
    ax.set_title(full_title, fontsize=14, pad=15)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


# Example usage:
plot_multiple_splines(PLOT_SPLINES, var_name="consumption_pd")
