"""
TODO:
- intercept-shift all curves s.t. they start at 0
- For anemia, divide domain & range by 12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rra_climate_health.data import DEFAULT_ROOT


SPLINE_ROOT = Path(DEFAULT_ROOT)
OUTPATH_ROOT = Path(DEFAULT_ROOT) / "plots"

VAR_DICT = {
    "consumption_pd": "Daily Consumption per capita",
    "days_over_30C_prev_0_mo": "Days over 30°C during birth month",
    "days_over_28C_prev_0_mo": "Days over 28°C during birth month",
    "days_over_28C_prev_3_mo_avg": "Average days over 28°C for 3 months prior to birth month",
    "days_over_28C_prev_9_mo_avg": "Average days over 28°C during 9 months before birth month",
}

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#d35400"]
CI_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#8e44ad", "#d35400"]


def intercept_shift(effect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shifts the 'effect' column of the DataFrame so that the first value is 0.
    This allows for better visual comparison of spline shapes across models.

    Parameters:
    - effect_df: DataFrame with columns ['value', 'effect', 'se'].

    Returns:
    - DataFrame with 'effect' column shifted to start at 0.
    """
    shifted_df = effect_df.copy()
    shifted_df["effect"] = shifted_df["effect"] - shifted_df["effect"].iloc[0]
    return shifted_df


def plot_multiple_splines(
    spline_dfs: list[tuple[Path, str]],  # Updated to use Path objects directly
    var_name: str,
    title: str | None = None,
    filepath: Path | None = None,  # Updated to use Path objects directly
    vmin: float | None = None,
    vmax: float | None = None,
    axes_labels=False,
):
    """
    Plots multiple SCAM spline effects on the same axes for comparison.

    Parameters:
    - spline_dfs: list of (DataFrame, label) tuples. Each DataFrame
      should have columns ['value', 'effect', 'se'].
    - var_name: Variable name (used for axis label lookup in VAR_DICT).
    - title: Optional plot title override.
    - filepath: If provided, save the figure to this path.
    """
    var_name_plt = VAR_DICT.get(var_name, var_name)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (effect_df, label) in enumerate(spline_dfs):
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
    if axes_labels:
        ax.set_xlabel(f"{var_name_plt}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Partial Effect (Log-Odds)", fontsize=12, fontweight="bold")

    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)

    if axes_labels:
        full_title = title if title else f"SCAM Spline Effect: {var_name_plt}"
        ax.set_title(full_title, fontsize=14, pad=15)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


# Update PLOT_SPLINES to use Path objects directly
PLOT_SPLINES = [
    (
        SPLINE_ROOT
        / "neonatal_mortality/models/2026_04_01.43/spline_effect_consumption_pd.parquet",
        "NMR Model using Avg Days over 28°C during 9 months prior to birth month",
    ),
    (
        SPLINE_ROOT
        / "neonatal_mortality/models/archive/run_2026_03_29/2026_03_29.08/spline_effect_consumption_pd.parquet",
        "NMR Model using Days over 30°C during birth month",
    ),
    (
        SPLINE_ROOT / "lbw/models/2026_03_30.139/spline_effect_ldi_pc_pd.parquet",
        "LBW model using Average days over 30°C during 9 months prior to birth month",
    ),
    (
        SPLINE_ROOT / "anemia/models/2026_04_03.14/spline_effect_ldi_pc_pd.parquet",
        "Anemia model using Avg Days over 30°C during survey year",
    ),
]

plot_splines_loaded = [(pd.read_parquet(p), l) for p, l in PLOT_SPLINES]
plot_splines_shifted = [
    (intercept_shift(df), label) for df, label in plot_splines_loaded
]

plot_multiple_splines(
    plot_splines_shifted,
    var_name="consumption_pd",
    filepath=OUTPATH_ROOT / "consumption_spline_comparison.png",
    # vmin=-1.75,
    # vmax=0.5,
)

PLOT_SPLINES = [
    (
        SPLINE_ROOT
        / "neonatal_mortality/models/2026_04_01.43/spline_effect_days_over_28C_prev_9_mo_avg.parquet",
        "NMR model using Avg Days over 28°C during 9 months prior to birth month",
    ),
    (
        SPLINE_ROOT
        / "neonatal_mortality/models/archive/run_2026_03_29/2026_03_29.08/spline_effect_days_over_30C_prev_0_mo.parquet",
        "NMR model using Days over 30°C during birth month",
    ),
    (
        SPLINE_ROOT
        / "lbw/models/2026_03_30.139/spline_effect_days_over_30C_past_9m.parquet",
        "LBW model using Avg Days over 30°C during 9 months prior to birth month",
    ),
    (
        SPLINE_ROOT / "anemia/models/2026_04_03.14/spline_effect_days_over_30C.parquet",
        "Anemia model using Avg Days over 30°C during survey year",
    ),
]

plot_splines_loaded = [(pd.read_parquet(p), l) for p, l in PLOT_SPLINES]

# For anemia, divide domain & range by 12
plot_splines_loaded[3][0]["value"] = plot_splines_loaded[3][0]["value"] / 12
# plot_splines_loaded[3][0]["effect"] = plot_splines_loaded[3][0]["effect"] / 12
# plot_splines_loaded[3][0]["se"] = plot_splines_loaded[3][0]["se"] / 12


plot_splines_shifted = [
    (intercept_shift(df), label) for df, label in plot_splines_loaded
]

plot_multiple_splines(
    plot_splines_shifted,
    var_name="Days over Threshold",
    filepath=OUTPATH_ROOT / "days_over_threshold_spline_comparison.png",
    # vmin=-1.75,
    # vmax=0.5,
)
