
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from typing import Any
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

    df["over_30"], o30_bins = pd.qcut(df.loc[df[threshold_varname] > 0, threshold_varname], 8, retbins=True)
    o30_bins = [0] + o30_bins
    df["ldi"], ldi_bins = pd.qcut(df.ldi_pc_pd, 10, retbins=True)

    x_ticks = range(len(o30_bins))
    x_labs = [f"{x:.1f}" for x in o30_bins]

    y_ticks = range(len(ldi_bins))
    y_labs = [f"{x:.1f}" for x in ldi_bins]

    vmin = 0
    vmax_dict = {'stunting': 0.5, 'wasting': 0.25, 'underweight': 0.40, 'anemia': 0.7, 'lbw':0.25}
    vmax = vmax_dict[measure]
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
    
    plot_gbd_comparison(merge_gbd_data(model_spec.measure, raw_df), 
        model_spec.measure, f"Model vs GBD for {model_spec.measure} {model_version} {submodel}",
        filepath=cm_data.models / model_version / f"gbd_comparison.png")

    plot_model_heatmaps(raw_df, model_spec.measure, 
                        filepath=cm_data.models / model_version / f"heatmap_comparison{submodel}.png")
    
