import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

ex_df = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_03_19.04/predictions_nnm_1_mo_do30_scam_summarywith_psu.parquet"
)

"""
df = ex_df.copy()
measure = "neonatal_mortality"
df.rename(columns={"child_mortality":"neonatal_mortality"},inplace=True)
df.rename(columns={"pred_fe":"no_re_fits"},inplace=True)
df.rename(columns={"pred_me":"fits"},inplace=True)
"""


def plot_model_heatmaps(df: str, measure: str, filepath: str = None) -> plt.Figure:  # type: ignore[name-defined]

    import seaborn as sns
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

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

    df[threshold_varname], o30_bins = pd.qcut(
        df.loc[df[threshold_varname] > 0, threshold_varname], 8, retbins=True
    )
    o30_bins = [0] + o30_bins
    if measure == "neonatal_mortality":
        df["ldi"], ldi_bins = pd.qcut(df["consumption_pd"], 10, retbins=True)
    else:
        df["ldi"], ldi_bins = pd.qcut(df.ldi_pc_pd, 10, retbins=True)

    x_ticks = range(len(o30_bins))
    x_labs = [f"{x:.1f}" for x in o30_bins]

    y_ticks = range(len(ldi_bins))
    y_labs = [f"{x:.1f}" for x in ldi_bins]

    vmin = 0
    vmax_dict = {
        "stunting": 0.5,
        "wasting": 0.25,
        "underweight": 0.40,
        "anemia": 0.7,
        "lbw": 0.25,
        "neonatal_mortality": 70,
    }
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

    axes[1].set_xlabel(
        threshold_varname.replace("_", " ").title(), fontsize=axislabel_size
    )
    axes[0].set_ylabel("Income", fontsize=axislabel_size)
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
    cb.set_label(f"{measure.capitalize()} Prevalence", size=axislabel_size)
    cb.ax.tick_params(labelsize=tick_size)
    cb.outline.set_edgecolor("none")

    # fig.suptitle(f"{measure.capitalize()} Prevalence", fontsize=20, y=1.02)
    if filepath:
        fig.savefig(filepath, dpi=300)
    else:
        fig.show()
