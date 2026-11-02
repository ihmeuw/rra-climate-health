
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from typing import Any
from decimal import Decimal, ROUND_UP
import matplotlib.pyplot as plt


from rra_climate_health.model_specification import ModelSpecification
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData


def get_scam_spline_effect(model, var_name, data_source, n_points=100, value_source=None):
    scam_lib = importr('scam')
    pandas2ri.activate()
    grid_df = data_source.iloc[[0]*n_points].reset_index(drop=True).copy()
    vmin, vmax = data_source[var_name].min(), data_source[var_name].max()
    grid_np = np.linspace(vmin, vmax, n_points)
    grid_df[var_name] = grid_np
    pred = scam_lib.predict_scam(model, newdata=grid_df, type='terms', se_fit=True)
    pandas2ri.deactivate()

    fit_matrix = np.array(pred.rx2('fit'))
    se_matrix = np.array(pred.rx2('se.fit'))
    col_names = list(r.colnames(pred.rx2('fit')))
    
    target_col = [i for i, name in enumerate(col_names) if f"s({var_name})" in name][0]

    if value_source is not None:
        real_min, real_max = value_source[var_name].min(), value_source[var_name].max()
        real_value_grid_np = np.linspace(real_min, real_max, n_points)

    return pd.DataFrame({
        'value': real_value_grid_np if value_source is not None else grid_np,
        'effect': fit_matrix[:, target_col],
        'se': se_matrix[:, target_col]
    })

def extract_knots_from_model(model, predictor_name):
    # Extract knots from the fitted model object
    pandas2ri.deactivate()
    smooth_terms = model.rx2('smooth')
    for i in range(len(smooth_terms)):
        term = smooth_terms[i]
        if term.rx2('term')[0] == predictor_name:
            return term.rx2('knots')
    raise ValueError(f"Predictor {predictor_name} not found in model smooth terms")

def plot_scam_spline(effect_df, original_data, var_name, knots = None, title=None, filepath=None):

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
    ax.plot(effect_df['value'], effect_df['effect'], color='#2c3e50', lw=2.5, label='Partial Effect')

    # 2. Add 95% Confidence Interval (1.96 * SE)
    lower_ci = effect_df['effect'] - (1.96 * effect_df['se'])
    upper_ci = effect_df['effect'] + (1.96 * effect_df['se'])
    
    ax.fill_between(effect_df['value'], lower_ci, upper_ci, 
                    color='#3498db', alpha=0.2, label='95% CI')

    # 3. Add the Rug Plot (The 'Rug' represents actual data distribution)
    # We place it at the very bottom of the current Y-axis
    obs = original_data[var_name].dropna()
    y_min = ax.get_ylim()[0]
    ax.plot(obs, np.full_like(obs, y_min), '|', color='black', 
            alpha=0.2, markersize=12, markeredgewidth=0.4)

    # 4. Reference line at 0 (No effect)
    ax.axhline(0, color='red', linestyle='--', alpha=0.4, lw=1)

    # 5. Add vertical lines for knots if provided
    if knots is not None:
        for knot in knots:
            ax.axvline(knot, color='green', linestyle=':', alpha=0.7, lw=1.5, label='Knot' if knot == knots[0] else None)

    # 5. Aesthetics
    ax.set_xlabel(f"{var_name}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Partial Effect (Log-Odds)", fontsize=12, fontweight='bold')
    
    full_title = title if title else f"SCAM Spline Effect: {var_name}"
    ax.set_title(full_title, fontsize=14, pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='best')

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


def merge_gbd_data(  # noqa: PLR0915
    measure: str,
    fitted_data: pd.DataFrame,
    fitted_column = "fits"
) -> plt.Figure:  # type: ignore[name-defined]
    from rra_climate_health.data_prep.location_mapping import FHS_HIERARCHY_PATH

    root = Path(DEFAULT_ROOT)
    fhs_loc_meta = (
        pd.read_parquet(FHS_HIERARCHY_PATH)
        .sort_values("sort_order")
        .reset_index(drop=True)
    )

    gbd = pd.read_parquet(root / "input" / "gbd_prevalence" / f"gbd_mean_{measure}_prevalence.parquet").reset_index()
    fitted_data = fitted_data.copy()
    #Align dtypes for merge
    for col in fitted_data.columns:
        if col in gbd.columns:
            fitted_data[col] = fitted_data[col].astype(gbd[col].dtype)

    gbd = gbd.merge(fhs_loc_meta[["ihme_loc_id", "location_id"]], on='location_id', how='left', validate='many_to_one')
    # take out location_id from index, add ihme_loc_id
    gbd = gbd.reset_index().set_index(["ihme_loc_id", "year_id", "age_group_id", "sex_id"])
    gbd = gbd.drop(columns=[x for x in gbd.columns if x not in ['gbd_mean_prevalence', 'location_id']], errors='ignore')
    gbd = gbd.rename(columns={'gbd_mean_prevalence': 'gbd'})

    prediction = fitted_data.rename(columns={fitted_column: "pred", "year_start":"year_id"}).groupby(
        ["ihme_loc_id", "year_id", "age_group_id", "sex_id"]).agg(
            {measure:'mean', 'pred':'mean'})
    plot_data = (
        prediction
        .join(gbd)
        # .dropna()
    )
    return plot_data

def plot_gbd_comparison(plot_data: pd.DataFrame, measure: str, title: str, filepath=None):
    rmse = np.sqrt(np.mean((plot_data['gbd'] - plot_data['pred']) ** 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_data['gbd'], plot_data['pred'], alpha=0.5)
    ax.plot([plot_data['gbd'].min(), plot_data['gbd'].max()], [plot_data['gbd'].min(), plot_data['gbd'].max()], 'r--')
    ax.set_xlabel('GBD Prevalence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Predicted Prevalence', fontsize=12, fontweight='bold')
    ax.set_title(title + f"\nRMSE: {rmse:.4f}", fontsize=14, pad=15)
    #Include RMSE as subtitle
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()

def ceil_1sig(x):
    d = Decimal(str(x))
    if d == 0:
        return d
    return d.quantize(Decimal(1).scaleb(d.adjusted()), rounding=ROUND_UP)

def plot_model_heatmaps(df: str, measure: str, filepath: str = None) -> plt.Figure:  # type: ignore[name-defined]
    import seaborn as sns
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    
    threshold_varname = next((x for x in df.columns if x.startswith("days_over")), None)
    if not threshold_varname:
        error_message = "No threshold variable found"
        raise ValueError(error_message)

    number_size = 16
    tick_size = 16
    axislabel_size = 18
    paneltitle_size = 18

    ldi_colname = "ldi_pc_pd" if "ldi_pc_pd" in df.columns else "consumption_pd"

    df["over_30"], o30_bins = pd.qcut(df.loc[df[threshold_varname] > 0, threshold_varname], 8, retbins=True, duplicates = 'drop')
    o30_bins = [0] + o30_bins
    df["ldi"], ldi_bins = pd.qcut(df[ldi_colname], 10, retbins=True, duplicates = 'drop')

    x_ticks = range(len(o30_bins))
    x_labs = [f"{x:.1f}" for x in o30_bins]

    y_ticks = range(len(ldi_bins))
    y_labs = [f"{x:.1f}" for x in ldi_bins]

    vmin_dict = {'stunting': 0, 'wasting': 0, 'underweight': 0, 'anemia': 0.4, 'lbw':0}
    vmax_dict = {'stunting': 0.5, 'wasting': 0.25, 'underweight': 0.40, 'anemia': 0.7, 'lbw':0.25}
    vmax = vmax_dict[measure] if measure in vmax_dict else ceil_1sig(df.groupby(['ihme_loc_id'])[measure].mean().max())
    vmin = vmin_dict[measure] if measure in vmax_dict else 0

    colorbin_interval = (vmax - vmin) / 10
    boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
    cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    fig, axes = plt.subplots(figsize=(24, 8), ncols=3)

    sns.heatmap(
        df.groupby(["ldi", "over_30"])[measure].mean().unstack(),
        ax=axes[0],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        #cbar_kws={"ticks": boundaries},
    )
    axes[0].set_title("Data", size=paneltitle_size)

    sns.heatmap(
        df.groupby(["ldi", "over_30"]).fits.mean().unstack(),
        ax=axes[1],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar = False,
        cbar_kws={"ticks": boundaries},
    )
    axes[1].set_title("With location random effects", size=paneltitle_size)

    sns.heatmap(
        df.groupby(["ldi", "over_30"]).no_re_fits.mean().unstack(),
        ax=axes[2],
        annot=True,
        fmt=".2f",
        annot_kws={"size": number_size, "weight": "regular"},
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        cbar = False,
        #cbar_kws={"ticks": boundaries},
    )
    axes[2].set_title("Without location random effects", size=paneltitle_size)

    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45, fontsize=tick_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, fontsize=tick_size)
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    axes[1].set_xlabel(threshold_varname.replace("_", " ").title(), fontsize=axislabel_size)
    axes[0].set_ylabel("Income", fontsize=axislabel_size)
    # axes[2].collections[0].colorbar.set_label(
    #     f"{measure.capitalize()} Prevalence", size=18
    # )
    #fig.tight_layout()

    # Adjust layout of the main heatmap subplots to make space for the colorbar
    # rect=[left, bottom, right, top] in figure coordinates
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Leave 10% on the right

    # Get the position of the last (or any) heatmap axis to align the colorbar
    # This must be done AFTER fig.tight_layout() has run
    last_ax_pos = axes[2].get_position()

    # Add a new axes for the colorbar
    # [left, bottom, width, height] in figure coordinates
    cbar_ax_left = last_ax_pos.x1 + 0.015  # Position to the right of the last heatmap, with a small gap
    cbar_ax_bottom = last_ax_pos.y0      # Align bottom with heatmap axes
    cbar_ax_width = 0.02                 # Width of the colorbar itself
    cbar_ax_height = last_ax_pos.height  # Align height with heatmap axes
    
    cbar_ax = fig.add_axes([cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height])

    # Create the colorbar in the new axes
    mappable = axes[0].collections[0] if axes[0].collections else axes[0].images[0]
    cb = fig.colorbar(mappable, cax=cbar_ax, norm=norm, ticks=boundaries)
    cb.set_label(f"{measure.capitalize()} Prevalence", size=axislabel_size)
    cb.ax.tick_params(labelsize=tick_size)
    cb.outline.set_edgecolor('none')

    #fig.suptitle(f"{measure.capitalize()} Prevalence", fontsize=20, y=1.02)
    if filepath:
        fig.savefig(filepath, dpi=300)
    else:
        fig.show()

# --- child_mortality heatmaps -----------------------------------------------
# The generic plot_model_heatmaps above bins on an LDI column and takes a plain
# mean of the measure. Neither works here: no child_mortality spec carries
# ldi_pc_pd or consumption_pd, and a plain mean of a per-age-interval death
# indicator is a per-interval hazard, not an under-5 quantity. The panels below
# divide by person-time instead, which is what the child_mortality figures in
# the other clone have always shown.

AGE_GROUP_DURATIONS: dict[str, int] = {
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
    0, 0.784781, 1.180789, 1.541445, 1.950251, 2.465952,
    3.103463, 4.003564, 5.541124, 9.413681, 112.879922,
]


def _measure_str(measure: Any) -> str:
    return measure.value if hasattr(measure, "value") else str(measure)


def cm_age_strata(
    model_spec: ModelSpecification,
    cm_data: ClimateMalnutritionData,
) -> tuple[list[str], str]:
    """Age dummies carried by the spec, and the reference stratum.

    Derived, not assumed: the strata actually present in the training data minus
    the ones the spec turns into dummies leaves exactly the reference cell. This
    is what lets the figure work for a model fit without neonates - there the
    data holds age_3_m..age_60_m and the spec names six dummies, so age_3_m is
    the reference. Hard-coding the full eight-interval set would both invent an
    age_1_m panel the model has no evidence for and silently drop every
    reference row (duration 0) from the other panels.
    """
    strata = cm_data.model_age_strata(model_spec.version.model)
    if strata is None:
        msg = f"{model_spec.measure} does not use named age strata"
        raise ValueError(msg)
    dummies = [p.name for p in model_spec.predictors if p.name in AGE_GROUP_DURATIONS]
    [reference] = [a for a in strata if a not in dummies]
    return dummies, reference


def _cm_duration_column(
    df: pd.DataFrame, dummies: list[str], reference: str
) -> "pd.Series[float]":
    """Months of exposure each row contributes.

    Rows with every dummy at 0 are the reference stratum and carry its duration;
    reading the duration off the dummies alone would give them 0 and drop them.
    """
    duration = pd.Series(0.0, index=df.index, dtype=float)
    on_a_dummy = pd.Series(False, index=df.index)
    for col in dummies:
        is_one = df[col].astype(int) == 1
        duration = duration + is_one * AGE_GROUP_DURATIONS[col]
        on_a_dummy = on_a_dummy | is_one
    duration = duration.where(on_a_dummy, AGE_GROUP_DURATIONS[reference])
    return duration


def _cm_person_time_rate(
    df: pd.DataFrame,
    value_col: str,
    x_col: str,
    y_col: str,
    x_bins: list,
    y_bins: list,
    duration: "pd.Series[float]",
) -> pd.DataFrame:
    """sum(value_col) / sum(duration) per (y_bin, x_bin)."""
    d = pd.DataFrame(
        {
            "v": df[value_col].to_numpy(),
            "_duration": duration.to_numpy(),
            "_x_bin": pd.cut(df[x_col], bins=x_bins, include_lowest=True, right=False),
            "_y_bin": pd.cut(df[y_col], bins=y_bins, include_lowest=True, right=False),
        }
    )
    d = d[d["_duration"] > 0]
    num = d.groupby(["_y_bin", "_x_bin"], observed=False)["v"].sum().unstack()
    denom = d.groupby(["_y_bin", "_x_bin"], observed=False)["_duration"].sum().unstack()
    return num / denom


def plot_model_heatmaps_child_mortality(
    raw_df: pd.DataFrame,
    dummies: list[str],
    reference: str,
    filepath: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    multiply_by: int = 1000,
) -> None:
    """Three panels - Data | With RE | Without RE - in deaths per 1000 person-months.

    All three come from raw_with_predictions (`child_mortality`, `fits`,
    `no_re_fits`), so the figure can be rebuilt after the fact without reloading
    the model pickle, and all three share one set of cells and one weighting.
    """
    import seaborn as sns
    import matplotlib.colors as mcolors

    threshold_varname = next(
        (c for c in raw_df.columns if c.startswith("days_over") or c.startswith("q")),
        None,
    )
    if not threshold_varname:
        msg = "No threshold variable found in raw_df columns"
        raise ValueError(msg)
    consumption_var = (
        "consumption_pd_cumul"
        if "consumption_pd_cumul" in raw_df.columns
        else "consumption_pd"
    )
    if consumption_var not in raw_df.columns:
        msg = f"no consumption column in raw_df; have {sorted(raw_df.columns)}"
        raise ValueError(msg)

    x_bins, y_bins = CHILD_MORTALITY_X_BINS, CHILD_MORTALITY_Y_BINS
    duration = _cm_duration_column(raw_df, dummies, reference)

    panels = [("Data", "child_mortality"), ("With RE", "fits"), ("Without RE", "no_re_fits")]
    mats = []
    for title, col in panels:
        if col not in raw_df.columns:
            msg = f"panel '{title}' needs column '{col}', absent from raw_df"
            raise ValueError(msg)
        mats.append(
            _cm_person_time_rate(
                raw_df, col, threshold_varname, consumption_var, x_bins, y_bins, duration
            )
            * multiply_by
        )

    # Scale from the data unless told otherwise. A fixed 1-7 was right while the
    # strata included age_1_m, whose hazard dominates; drop neonates and the
    # whole surface falls under 3, so a fixed floor of 1 would flatten every
    # cell to one colour.
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(ceil_1sig(float(np.nanmax(mats[0].to_numpy()))))
    colorbin_interval = (vmax - vmin) / 10
    boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
    cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    number_size, tick_size, axislabel_size, paneltitle_size = 16, 16, 18, 18
    fig, axes = plt.subplots(figsize=(24, 8), ncols=3)
    for ax, (title, _), mat in zip(axes, panels, mats):
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".1f", annot_kws={"size": number_size},
            cmap=cmap, norm=norm, cbar=False, linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, fontsize=paneltitle_size)
        ax.set_xticks(range(len(x_bins)))
        ax.set_xticklabels([f"{b:.1f}" for b in x_bins], fontsize=tick_size, rotation=0)
        ax.set_yticks(range(len(y_bins)))
        ax.set_yticklabels([f"{b:.1f}" for b in y_bins], fontsize=tick_size, rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[1].set_xlabel(threshold_varname.replace("_", " ").title(), fontsize=axislabel_size)
    axes[0].set_ylabel(consumption_var.replace("_", " ").title(), fontsize=axislabel_size)

    fig.tight_layout(rect=[0, 0, 0.9, 0.94])
    pos = axes[2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height])
    mappable = axes[0].collections[0] if axes[0].collections else axes[0].images[0]
    cb = fig.colorbar(mappable, cax=cbar_ax, norm=norm, ticks=boundaries)
    cb.set_label("Deaths per 1000 person-months", size=axislabel_size)
    cb.ax.tick_params(labelsize=tick_size)
    cb.outline.set_edgecolor("none")

    strata = ", ".join([reference, *dummies])
    fig.suptitle(f"child_mortality, strata: {strata}", fontsize=14, y=0.98)
    if filepath:
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
    else:
        fig.show()


def run_training_diagnostics(model: Any, 
                                df: pd.DataFrame, 
                                model_spec: ModelSpecification, 
                                cm_data: ClimateMalnutritionData, 
                                model_version: str, 
                                submodel: list[tuple[str, str]] | None,
                                raw_df: pd.DataFrame,
                                var_info: dict[str, Any]) -> None:
    if not submodel:
        submodel = ""
    for var in model_spec.predictors:
        if var.spline:
            effect_df = get_scam_spline_effect(model, var.name, df, value_source=raw_df)
            knots = extract_knots_from_model(model, var.name)
            if var.transform and var.transform.type == 'scaling':
                transformed_knots = var_info[var.name]['transformer'].inverse_transform(knots)
            else:
                transformed_knots = knots
            effect_df.to_parquet(cm_data.models / model_version / f"spline_effect_{var.name}{submodel}.parquet")
            plot_scam_spline(effect_df, raw_df, var.name, knots = transformed_knots,
                            title=f"Spline Effect for {var.name} submodel {submodel}",
                            filepath=cm_data.models / model_version / f"spline_effect_{var.name}{submodel}.png")
    
    # plot_gbd_comparison(merge_gbd_data(model_spec.measure, raw_df), 
    #     model_spec.measure, f"Model vs GBD for {model_spec.measure} {model_version} {submodel}",
    #     filepath=cm_data.models / model_version / f"gbd_comparison.png")

    # A diagnostic must not condemn a fit that is already saved. Workflow 639321
    # produced a complete model and still went FATAL because the heatmap raised,
    # so the failure is reported to stderr and the task is allowed to succeed.
    heatmap_path = cm_data.models / model_version / f"heatmap_comparison{submodel}.png"
    try:
        if _measure_str(model_spec.measure) == "child_mortality":
            dummies, reference = cm_age_strata(model_spec, cm_data)
            plot_model_heatmaps_child_mortality(
                raw_df, dummies, reference, filepath=heatmap_path
            )
        else:
            plot_model_heatmaps(raw_df, model_spec.measure, filepath=heatmap_path)
    except Exception:
        import traceback
        print(f"WARNING: heatmap diagnostic failed; {heatmap_path} not written")
        traceback.print_exc()
    
