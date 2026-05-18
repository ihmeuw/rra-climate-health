import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rpy2.robjects import pandas2ri, r, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from typing import Any
import matplotlib.pyplot as plt

from rra_climate_health.model_specification import ModelSpecification
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData


def get_scam_spline_effect(
    model, var_name, data_source, n_points=100, value_source=None
):
    scam_lib = importr("scam")
    grid_df = data_source.iloc[[0] * n_points].reset_index(drop=True).copy()
    vmin, vmax = data_source[var_name].min(), data_source[var_name].max()
    grid_np = np.linspace(vmin, vmax, n_points)
    grid_df[var_name] = grid_np
    with localconverter(default_converter + pandas2ri.converter):
        r_grid_df = pandas2ri.py2rpy(grid_df)
    pred = scam_lib.predict_scam(model, newdata=r_grid_df, type="terms", se_fit=True)

    fit_matrix = np.array(pred.rx2("fit"))
    se_matrix = np.array(pred.rx2("se.fit"))
    col_names = list(r.colnames(pred.rx2("fit")))

    target_col = [i for i, name in enumerate(col_names) if f"s({var_name})" in name][0]

    if value_source is not None:
        real_min, real_max = value_source[var_name].min(), value_source[var_name].max()
        real_value_grid_np = np.linspace(real_min, real_max, n_points)

    return pd.DataFrame(
        {
            "value": real_value_grid_np if value_source is not None else grid_np,
            "effect": fit_matrix[:, target_col],
            "se": se_matrix[:, target_col],
        }
    )


def extract_knots_from_model(model, predictor_name):
    # Extract knots from the fitted model object
    smooth_terms = model.rx2("smooth")
    for i in range(len(smooth_terms)):
        term = smooth_terms[i]
        if term.rx2("term")[0] == predictor_name:
            return term.rx2("knots")
    raise ValueError(f"Predictor {predictor_name} not found in model smooth terms")


def plot_scam_spline(
    effect_df, original_data, var_name, knots=None, title=None, filepath=None
):
    """
    Plots a SCAM spline effect with 95% CI and a rug plot.

    Parameters:
    - effect_df: DataFrame with ['value', 'effect', 'se']
    - original_data: The full DataFrame (used for the rug plot)
    - var_name: Name of the variable being plotted (e.g., 'temperature')
    - knots: List of knot values (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot the Partial Effect (Spline)
    ax.plot(
        effect_df["value"],
        effect_df["effect"],
        color="#2c3e50",
        lw=2.5,
        label="Partial Effect",
    )

    # 2. Add 95% Confidence Interval (1.96 * SE)
    lower_ci = effect_df["effect"] - (1.96 * effect_df["se"])
    upper_ci = effect_df["effect"] + (1.96 * effect_df["se"])

    ax.fill_between(
        effect_df["value"],
        lower_ci,
        upper_ci,
        color="#3498db",
        alpha=0.2,
        label="95% CI",
    )

    # 3. Add the Rug Plot (The 'Rug' represents actual data distribution)
    # We place it at the very bottom of the current Y-axis
    obs = original_data[var_name].dropna()
    y_min = ax.get_ylim()[0]
    ax.plot(
        obs,
        np.full_like(obs, y_min),
        "|",
        color="black",
        alpha=0.2,
        markersize=12,
        markeredgewidth=0.4,
    )

    # 4. Reference line at 0 (No effect)
    ax.axhline(0, color="red", linestyle="--", alpha=0.4, lw=1)

    # 5. Add vertical lines for knots if provided
    if knots is not None:
        for knot in knots:
            ax.axvline(
                knot,
                color="green",
                linestyle=":",
                alpha=0.7,
                lw=1.5,
                label="Knot" if knot == knots[0] else None,
            )

    # 5. Aesthetics
    ax.set_xlabel(f"{var_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Partial Effect (Log-Odds)", fontsize=12, fontweight="bold")

    full_title = title if title else f"SCAM Spline Effect: {var_name}"
    ax.set_title(full_title, fontsize=14, pad=15)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


def plot_scam_spline_presentation(
    effect_df, original_data, var_name, knots=None, title=None, filepath=None
):
    """
    Plots a SCAM spline effect with 95% CI and a rug plot.

    Parameters:
    - effect_df: DataFrame with ['value', 'effect', 'se']
    - original_data: The full DataFrame (used for the rug plot)
    - var_name: Name of the variable being plotted (e.g., 'temperature')
    - knots: List of knot values (optional)
    """
    var_dict = {
        "consumption_pd": "Daily Consumption per capita",
        "days_over_30C_prev_0_mo": "Days over 30°C during birth month",
        "days_over_28C_prev_0_mo": "Days over 28°C during birth month",
        "days_over_28C_prev_3_mo_avg": "Average number of days over 28°C for 3 months prior to birth month",
    }
    if var_name in var_dict:
        var_name_plt = var_dict[var_name]
    else:
        var_name_plt = var_name

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot the Partial Effect (Spline)
    ax.plot(
        effect_df["value"],
        effect_df["effect"],
        color="#2c3e50",
        lw=2.5,
        label="Partial Effect",
    )

    # 2. Add 95% Confidence Interval (1.96 * SE)
    lower_ci = effect_df["effect"] - (1.96 * effect_df["se"])
    upper_ci = effect_df["effect"] + (1.96 * effect_df["se"])

    ax.fill_between(
        effect_df["value"],
        lower_ci,
        upper_ci,
        color="#3498db",
        alpha=0.2,
        label="95% CI",
    )

    # 3. Add the Rug Plot (The 'Rug' represents actual data distribution)
    # We place it at the very bottom of the current Y-axis
    obs = original_data[var_name].dropna()
    y_min = ax.get_ylim()[0]
    # ax.plot(
    #     obs,
    #     np.full_like(obs, y_min),
    #     "|",
    #     color="black",
    #     alpha=0.2,
    #     markersize=12,
    #     markeredgewidth=0.4,
    # )

    # 4. Reference line at 0 (No effect)
    # ax.axhline(0, color="red", linestyle="--", alpha=0.4, lw=1)

    # 5. Add vertical lines for knots if provided
    # if knots is not None:
    #     for knot in knots:
    #         ax.axvline(
    #             knot,
    #             color="green",
    #             linestyle=":",
    #             alpha=0.7,
    #             lw=1.5,
    #             label="Knot" if knot == knots[0] else None,
    #         )

    # 5. Aesthetics
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


def merge_gbd_data(  # noqa: PLR0915
    measure: str, fitted_data: pd.DataFrame, fitted_column="fits"
) -> plt.Figure:  # type: ignore[name-defined]
    from rra_climate_health.data_prep.location_mapping import FHS_HIERARCHY_PATH

    root = Path(DEFAULT_ROOT)
    fhs_loc_meta = (
        pd.read_parquet(FHS_HIERARCHY_PATH)
        .sort_values("sort_order")
        .reset_index(drop=True)
    )

    gbd = pd.read_parquet(
        root / "input" / "gbd_prevalence" / f"gbd_mean_{measure}_prevalence.parquet"
    ).reset_index()
    fitted_data = fitted_data.copy()
    # Align dtypes for merge
    for col in fitted_data.columns:
        if col in gbd.columns:
            fitted_data[col] = fitted_data[col].astype(gbd[col].dtype)

    gbd = gbd.merge(
        fhs_loc_meta[["ihme_loc_id", "location_id"]],
        on="location_id",
        how="left",
        validate="many_to_one",
    )
    # take out location_id from index, add ihme_loc_id
    if measure == "neonatal_mortality":
        gbd = gbd.reset_index().set_index(["ihme_loc_id", "year_id", "sex_id"])
    else:
        gbd = gbd.reset_index().set_index(
            ["ihme_loc_id", "year_id", "age_group_id", "sex_id"]
        )
    gbd = gbd.drop(
        columns=[
            x for x in gbd.columns if x not in ["gbd_mean_prevalence", "location_id"]
        ],
        errors="ignore",
    )
    gbd = gbd.rename(columns={"gbd_mean_prevalence": "gbd"})

    if measure == "neonatal_mortality":
        prediction = (
            fitted_data.rename(columns={fitted_column: "pred", "birth_year": "year_id"})
            .groupby(["ihme_loc_id", "year_id", "sex_id"])
            .agg({measure: "mean", "pred": "mean"})
        )
    elif measure == "child_mortality":
        prediction = (
            fitted_data.rename(columns={fitted_column: "pred", "int_year": "year_id"})
            .groupby(["ihme_loc_id", "year_id", "sex_id"])
            .agg({measure: "mean", "pred": "mean"})
        )
    else:
        prediction = (
            fitted_data.rename(columns={fitted_column: "pred", "year_start": "year_id"})
            .groupby(["ihme_loc_id", "year_id", "age_group_id", "sex_id"])
            .agg({measure: "mean", "pred": "mean"})
        )
    plot_data = (
        prediction.join(gbd)
        # .dropna()
    )
    return plot_data


def plot_gbd_comparison(
    plot_data: pd.DataFrame, measure: str, title: str, filepath=None
):
    rmse = np.sqrt(np.mean((plot_data["gbd"] - plot_data["pred"]) ** 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_data["gbd"], plot_data["pred"], alpha=0.5)
    ax.plot(
        [plot_data["gbd"].min(), plot_data["gbd"].max()],
        [plot_data["gbd"].min(), plot_data["gbd"].max()],
        "r--",
    )
    ax.set_xlabel("GBD Prevalence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model Predicted Prevalence", fontsize=12, fontweight="bold")
    ax.set_title(title + f"\nRMSE: {rmse:.4f}", fontsize=14, pad=15)
    # Include RMSE as subtitle
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


AGE_GROUP_DURATIONS = {
    "age_1_m": 1,
    "age_3_m": 2,
    "age_6_m": 3,
    "age_12_m": 6,
    "age_24_m": 12,
    "age_36_m": 12,
    "age_48_m": 12,
    "age_60_m": 12,
}

CHILD_MORTALITY_X_BINS = [0, 0.1, 2, 4, 9, 31]
CHILD_MORTALITY_Y_BINS = [
    0,
    0.784781,
    1.180789,
    1.541445,
    1.950251,
    2.465952,
    3.103463,
    4.003564,
    5.541124,
    9.413681,
    112.879922,
]


def _measure_str(measure: Any) -> str:
    return measure.value if hasattr(measure, "value") else str(measure)


def _age_duration_column(df: pd.DataFrame) -> pd.Series:
    duration = pd.Series(0, index=df.index, dtype=float)
    for col, months in AGE_GROUP_DURATIONS.items():
        if col in df.columns:
            duration = duration + (df[col].astype(int) == 1) * months
    return duration


def _person_time_rate(
    df: pd.DataFrame,
    value_col: str,
    x_col: str,
    y_col: str,
    x_bins: list,
    y_bins: list,
) -> pd.DataFrame:
    """Sum(value_col) / sum(age-interval duration) per (y_bin, x_bin)."""
    d = df.copy()
    d["_duration"] = _age_duration_column(d)
    d = d[d["_duration"] > 0]
    d["_x_bin"] = pd.cut(d[x_col], bins=x_bins, include_lowest=True, right=False)
    d["_y_bin"] = pd.cut(d[y_col], bins=y_bins, include_lowest=True, right=False)
    num = d.groupby(["_y_bin", "_x_bin"], observed=False)[value_col].sum().unstack()
    denom = d.groupby(["_y_bin", "_x_bin"], observed=False)["_duration"].sum().unstack()
    return num / denom


def _synthetic_no_re_grid_cm(
    model: Any,
    df: pd.DataFrame,
    climate_var: str,
    consumption_var: str,
    n_points: int = 500,
) -> pd.DataFrame:
    """Build a (climate x consumption x 8 age-intervals) grid and predict with
    `exclude='s(ihme_loc_id)'`, mirroring the R workflow that produces
    `_predictions_with_both_splines_ranged_all_ages.parquet`. `df` is the
    processed model dataframe so categorical levels match the fitted model.
    """
    age_vars = list(AGE_GROUP_DURATIONS.keys())

    climate_grid = np.linspace(
        float(df[climate_var].min()), float(df[climate_var].max()), n_points
    )
    cons_grid = np.linspace(
        float(df[consumption_var].min()), float(df[consumption_var].max()), n_points
    )
    cx, cy = np.meshgrid(climate_grid, cons_grid)
    base = pd.DataFrame({climate_var: cx.ravel(), consumption_var: cy.ravel()})

    skip_cols = {climate_var, consumption_var, *age_vars}
    # For constant-valued categorical columns, pandas2ri can convert to an R
    # factor that only declares the observed level. Track them so we can
    # reconstruct the factor in R with all training-time levels.
    constant_factor_cols: dict[str, tuple[Any, list]] = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_categorical_dtype(df[col]):
            mode_val = df[col].mode().iloc[0]
            levels = list(df[col].cat.categories)
            base[col] = pd.Categorical([mode_val] * len(base), categories=levels)
            constant_factor_cols[col] = (mode_val, levels)
        elif pd.api.types.is_numeric_dtype(df[col]):
            base[col] = float(df[col].median())
        elif pd.api.types.is_object_dtype(df[col]):
            base[col] = df[col].mode().iloc[0]

    def _dummy_level(target: int, original_col: pd.Series) -> Any:
        """Map int 0/1 to the matching level of a (possibly categorical) column."""
        if pd.api.types.is_categorical_dtype(original_col):
            for lv in original_col.cat.categories:
                try:
                    if int(lv) == target:
                        return lv
                except (ValueError, TypeError):
                    if str(lv) == str(target):
                        return lv
            raise ValueError(
                f"Cannot map {target} to categories {list(original_col.cat.categories)}"
            )
        return target

    def _assign_dummy(g: pd.DataFrame, col: str, target: int) -> None:
        if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
            level = _dummy_level(target, df[col])
            g[col] = pd.Categorical(
                [level] * len(g), categories=df[col].cat.categories
            )
        else:
            g[col] = target

    grids = []
    for av in age_vars:
        g = base.copy()
        for other in age_vars:
            _assign_dummy(g, other, 0)
        _assign_dummy(g, av, 1)
        g["_age_group"] = av
        g["_duration"] = AGE_GROUP_DURATIONS[av]
        grids.append(g)
    grid = pd.concat(grids, ignore_index=True)
    # pd.concat preserves Categorical dtype only when all chunks agree; re-cast
    # defensively so pandas2ri receives a 2-level factor for each age dummy.
    for av in age_vars:
        if av in df.columns and pd.api.types.is_categorical_dtype(df[av]):
            grid[av] = pd.Categorical(
                grid[av], categories=df[av].cat.categories
            )

    scam_lib = importr("scam")
    grid_for_r = grid.drop(columns=["_age_group", "_duration"])
    with localconverter(default_converter + pandas2ri.converter):
        r_grid = pandas2ri.py2rpy(grid_for_r)

    from rpy2 import robjects

    robjects.globalenv["r_grid_synth"] = r_grid
    for col, (mode_val, levels) in constant_factor_cols.items():
        levels_str = ", ".join(f'"{lv}"' for lv in levels)
        robjects.r(
            f'r_grid_synth${col} <- factor(rep("{mode_val}", nrow(r_grid_synth)), '
            f"levels = c({levels_str}))"
        )
    r_grid = robjects.globalenv["r_grid_synth"]

    pred = scam_lib.predict_scam(
        model, newdata=r_grid, type="response", exclude="s(ihme_loc_id)"
    )
    grid["pred_prob_fe"] = np.array(pred)
    return grid


def plot_model_heatmaps_child_mortality(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    model: Any,
    filepath: str | None = None,
    vmin: float = 1.0,
    vmax: float = 7.0,
    multiply_by: int = 1000,
) -> None:
    """Three-panel heatmap (Data | With RE | Without RE) for child_mortality.

    Aggregation is sum(value) / sum(age-interval duration) per cell, multiplied
    by 1000 -> deaths per 1000 person-months. The "Without RE" panel uses a
    500x500x8 synthetic grid (8 age intervals per cell) so the cell value is
    implicitly an under-5 quantity, comparable to gbd_mean child_mortality.
    """
    import seaborn as sns
    import matplotlib.colors as mcolors

    threshold_varname = next(
        (
            x
            for x in raw_df.columns
            if x.startswith("days_over") or x.startswith("q")
        ),
        None,
    )
    if not threshold_varname:
        raise ValueError("No threshold variable found in raw_df columns")
    consumption_var = (
        "consumption_pd_cumul"
        if "consumption_pd_cumul" in raw_df.columns
        else "consumption_pd"
    )

    x_bins = CHILD_MORTALITY_X_BINS
    y_bins = CHILD_MORTALITY_Y_BINS

    data_hm = (
        _person_time_rate(
            raw_df, "child_mortality", threshold_varname, consumption_var, x_bins, y_bins
        )
        * multiply_by
    )
    re_hm = (
        _person_time_rate(
            raw_df, "fits", threshold_varname, consumption_var, x_bins, y_bins
        )
        * multiply_by
    )

    grid = _synthetic_no_re_grid_cm(
        model, df, threshold_varname, consumption_var, n_points=500
    )
    grid["_x_bin"] = pd.cut(
        grid[threshold_varname], bins=x_bins, include_lowest=True, right=False
    )
    grid["_y_bin"] = pd.cut(
        grid[consumption_var], bins=y_bins, include_lowest=True, right=False
    )
    num = (
        grid.groupby(["_y_bin", "_x_bin"], observed=False)["pred_prob_fe"]
        .sum()
        .unstack()
    )
    denom = (
        grid.groupby(["_y_bin", "_x_bin"], observed=False)["_duration"].sum().unstack()
    )
    no_re_hm = (num / denom) * multiply_by

    colorbin_interval = (vmax - vmin) / 10
    boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
    cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    x_labs = [f"{b:.1f}" for b in x_bins]
    y_labs = [f"{b:.1f}" for b in y_bins]

    fig, axes = plt.subplots(figsize=(24, 8), ncols=3)
    panels = [
        (data_hm, "Data"),
        (re_hm, "With location random effects"),
        (no_re_hm, "Without location random effects"),
    ]
    for ax, (hm, title) in zip(axes, panels):
        sns.heatmap(
            hm,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 14, "weight": "regular"},
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
        )
        ax.set_title(title, size=18)
        ax.set_xticks(range(len(x_labs)))
        ax.set_xticklabels(x_labs, rotation=45, fontsize=14)
        ax.set_yticks(range(len(y_labs)))
        ax.set_yticklabels(y_labs, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if threshold_varname.startswith("days_over_30C"):
        axes[1].set_xlabel("Days over 30°C (monthly cumulative)", fontsize=18)
    elif threshold_varname.startswith("days_over"):
        thresh = threshold_varname.split("_")[2]
        axes[1].set_xlabel(f"Days over {thresh} (monthly cumulative)", fontsize=18)
    elif threshold_varname.startswith("q"):
        axes[1].set_xlabel(
            f"{threshold_varname.replace('_', ' ')} (cumulative days over percentile)",
            fontsize=18,
        )
    else:
        axes[1].set_xlabel(
            threshold_varname.replace("_", " ").title(), fontsize=18
        )
    axes[0].set_ylabel("Daily consumption per capita (cumulative)", fontsize=18)

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    last_ax_pos = axes[2].get_position()
    cbar_ax = fig.add_axes(
        [last_ax_pos.x1 + 0.015, last_ax_pos.y0, 0.02, last_ax_pos.height]
    )
    mappable = axes[0].collections[0] if axes[0].collections else axes[0].images[0]
    cb = fig.colorbar(mappable, cax=cbar_ax, norm=norm, ticks=boundaries)
    cb.set_label("Child mortality (per 1000 person-months)", size=18)
    cb.ax.tick_params(labelsize=14)
    cb.outline.set_edgecolor("none")

    if filepath:
        fig.savefig(filepath, dpi=300)
    else:
        fig.show()
    plt.close(fig)


def plot_model_heatmaps(df: str, measure: str, filepath: str = None) -> plt.Figure:  # type: ignore[name-defined]

    import seaborn as sns
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    df = df.copy()

    threshold_varname = next(
        (x for x in df.columns if (x.startswith("days_over")) | x.startswith("q")), None
    )
    if not threshold_varname:
        error_message = "No threshold variable found"
        raise ValueError(error_message)

    number_size = 16
    tick_size = 16
    axislabel_size = 18
    paneltitle_size = 18

    custom_x_bins = [
        1,
        3,
        8,
        15,
        22,
        31,
    ]

    custom_y_bins = [
        0,
        0.784781,
        1.180789,
        1.541445,
        1.950251,
        2.465952,
        3.103463,
        4.003564,
        5.541124,
        9.413681,
        112.879922,
    ]

    if measure == "neonatal_mortality":
        df[threshold_varname], o30_bins = pd.cut(
            df.loc[df[threshold_varname] > 0, threshold_varname],
            bins=custom_x_bins,
            include_lowest=True,
            right=False,
            retbins=True,
        )
        # df[threshold_varname], o30_bins = pd.qcut(
        #     df.loc[df[threshold_varname] > 0, threshold_varname], 5, retbins=True
        # )
        # o30_bins = [0] + list(o30_bins)
    else:
        df[threshold_varname], o30_bins = pd.qcut(
            df.loc[df[threshold_varname] > 0, threshold_varname], 8, retbins=True
        )
        o30_bins = [0] + o30_bins
    if measure == "neonatal_mortality":
        df["ldi"], ldi_bins = pd.cut(
            df["consumption_pd"],
            bins=custom_y_bins,
            include_lowest=True,
            right=False,
            retbins=True,
        )
    else:
        df["ldi"], ldi_bins = pd.qcut(df.ldi_pc_pd, 10, retbins=True)

    x_ticks = range(len(o30_bins))
    x_labs = [f"{x:.1f}" for x in o30_bins]

    y_ticks = range(len(ldi_bins))
    y_labs = [f"{x:.1f}" for x in ldi_bins]
    # y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(ldi_bins))]

    vmin = 0
    vmin_dict = {
        "stunting": 0,
        "wasting": 0,
        "underweight": 0,
        "anemia": 0,
        "lbw": 0,
        "neonatal_mortality": 20,
    }
    vmax_dict = {
        "stunting": 0.5,
        "wasting": 0.25,
        "underweight": 0.40,
        "anemia": 0.7,
        "lbw": 0.25,
        "neonatal_mortality": 50,
    }
    vmin = vmin_dict[measure]
    vmax = vmax_dict[measure]
    colorbin_interval = (vmax - vmin) / 10
    boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
    cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    if measure == "neonatal_mortality":
        df[measure] = df[measure] * 1000  # convert to per 1000 live births
        df["fits"] = df["fits"] * 1000
        df["no_re_fits"] = df["no_re_fits"] * 1000

    fig, axes = plt.subplots(figsize=(24, 8), ncols=3)

    sns.heatmap(
        df.groupby(["ldi", threshold_varname])[measure].mean().unstack(),
        ax=axes[0],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        # cbar_kws={"ticks": boundaries},
    )
    axes[0].set_title("Data", size=paneltitle_size)

    sns.heatmap(
        df.groupby(["ldi", threshold_varname]).fits.mean().unstack(),
        ax=axes[1],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cbar_kws={"ticks": boundaries},
    )
    axes[1].set_title("With location random effects", size=paneltitle_size)

    sns.heatmap(
        df.groupby(["ldi", threshold_varname]).no_re_fits.mean().unstack(),
        ax=axes[2],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        # cbar_kws={"ticks": boundaries},
    )
    axes[2].set_title("Without location random effects", size=paneltitle_size)

    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45, fontsize=tick_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, fontsize=tick_size)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if threshold_varname == "days_over_30C_prev_0_mo":
        axes[1].set_xlabel("Days over 30°C during birth month", fontsize=axislabel_size)
    elif threshold_varname == "days_over_28C_prev_0_mo":
        axes[1].set_xlabel("Days over 28°C during birth month", fontsize=axislabel_size)
    elif threshold_varname == "days_over_28C_prev_3_mo_avg":
        axes[1].set_xlabel(
            "Average number of days over 28°C for 3 months prior to birth month",
            fontsize=axislabel_size,
        )
    else:
        axes[1].set_xlabel(
            threshold_varname.replace("_", " ").title(), fontsize=axislabel_size
        )
    axes[0].set_ylabel("Daily Consumption per capita", fontsize=axislabel_size)
    # axes[2].collections[0].colorbar.set_label(
    #     f"{measure.capitalize()} Prevalence", size=18
    # )
    # fig.tight_layout()

    # Adjust layout of the main heatmap subplots to make space for the colorbar
    # rect=[left, bottom, right, top] in figure coordinates
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave 10% on the right

    # Get the position of the last (or any) heatmap axis to align the colorbar
    # This must be done AFTER fig.tight_layout() has run
    last_ax_pos = axes[2].get_position()

    # Add a new axes for the colorbar
    # [left, bottom, width, height] in figure coordinates
    cbar_ax_left = (
        last_ax_pos.x1 + 0.015
    )  # Position to the right of the last heatmap, with a small gap
    cbar_ax_bottom = last_ax_pos.y0  # Align bottom with heatmap axes
    cbar_ax_width = 0.02  # Width of the colorbar itself
    cbar_ax_height = last_ax_pos.height  # Align height with heatmap axes

    cbar_ax = fig.add_axes(
        [cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height]
    )

    # Create the colorbar in the new axes
    mappable = axes[0].collections[0] if axes[0].collections else axes[0].images[0]
    cb = fig.colorbar(mappable, cax=cbar_ax, norm=norm, ticks=boundaries)
    cb.set_label(f"{measure.replace("_"," ").capitalize()}", size=axislabel_size)
    cb.ax.tick_params(labelsize=tick_size)
    cb.outline.set_edgecolor("none")

    # fig.suptitle(f"{measure.capitalize()} Prevalence", fontsize=20, y=1.02)
    if filepath:
        fig.savefig(filepath, dpi=300)
    else:
        fig.show()


def run_training_diagnostics(
    model: Any,
    df: pd.DataFrame,
    model_spec: ModelSpecification,
    cm_data: ClimateMalnutritionData,
    model_version: str,
    submodel: list[tuple[str, str]] | None,
    raw_df: pd.DataFrame,
    var_info: dict[str, Any],
) -> None:
    if not submodel:
        submodel = ""
    for var in model_spec.predictors:
        if var.spline:
            effect_df = get_scam_spline_effect(model, var.name, df, value_source=raw_df)
            knots = extract_knots_from_model(model, var.name)
            if var.transform and var.transform.type == "scaling":
                transformed_knots = var_info[var.name]["transformer"].inverse_transform(
                    knots
                )
            else:
                transformed_knots = knots
            effect_df.to_parquet(
                cm_data.models
                / model_version
                / f"spline_effect_{var.name}{submodel}.parquet"
            )
            plot_scam_spline(
                effect_df,
                raw_df,
                var.name,
                knots=transformed_knots,
                title=f"Spline Effect for {var.name} submodel {submodel}",
                filepath=cm_data.models
                / model_version
                / f"spline_effect_{var.name}{submodel}.png",
            )

            plot_scam_spline_presentation(
                effect_df,
                raw_df,
                var.name,
                knots=transformed_knots,
                # title=f"Spline Effect for {var.name}",
                filepath=cm_data.models
                / model_version
                / f"spline_effect_{var.name}{submodel}_ppt.png",
            )
    # save predictions to raw data:
    raw_df.to_parquet(cm_data.models / model_version / f"predictions{submodel}.parquet")

    plot_gbd_comparison(
        merge_gbd_data(model_spec.measure, raw_df),
        model_spec.measure,
        f"Model vs GBD for {model_spec.measure} {model_version} {submodel}",
        filepath=cm_data.models / model_version / f"gbd_comparison.png",
    )

    if _measure_str(model_spec.measure) == "child_mortality":
        plot_model_heatmaps_child_mortality(
            raw_df=raw_df,
            df=df,
            model=model,
            filepath=cm_data.models / model_version / f"heatmap_comparison{submodel}.png",
        )
    else:
        plot_model_heatmaps(
            raw_df,
            model_spec.measure,
            filepath=cm_data.models / model_version / f"heatmap_comparison{submodel}.png",
        )
