"""Diagnostic plots and tables for the residual step.

Ported from ``malnutrition_fhs/src/plotting.py``.  ``make_results_plots`` is the
entry point used by the residual step; it produces

* ``{measure}_prevalence_plots.pdf`` -- one page per location, a sex-by-age grid
  of GBD / original model / residual model / shifted model prevalence,
* ``{measure}_SEV_plots.pdf`` -- the same grid for SEVs,
* ``superregion_prev.pdf`` -- global and super-region prevalence by scenario,
* ``table1_cumulative_case_counts.csv`` -- cumulative case count differences
  between scenario pairs.
"""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from rra_climate_health.constants import (
    FIRST_FORECAST_YEAR,
    REFERENCE_SCENARIO,
    VALID_AGE_GROUPS_FOR_MEASURE,
)
from rra_climate_health.data import ClimateMalnutritionData
from rra_climate_health.inference.run_inference import load_population_timeseries
from rra_climate_health.residual.residual_data import (
    aggregate_forecast_hierarchy,
    get_gbd_data,
)

LABEL_MAP = {
    "gbd_value": "GBD Prevalence",
    "adjusted_model_value": "Shifted model",
    "adjusted_ssp585": "Shifted SSP 5-8.5",
    "adjusted_ssp126": "Shifted SSP 1-2.6",
    "adjusted_constant_climate": "Shifted constant climate",
    "model_value": "Original model",
    "residual_model_value": "Residual model",
    "gbd_sev": "GBD SEV",
    "adjusted_sev_val": "Shifted SEV",
    "adjusted_sev_ssp585": "Shifted SEV SSP 5-8.5",
    "adjusted_sev_ssp126": "Shifted SEV SSP 1-2.6",
    "adjusted_sev_constant_climate": "Shifted SEV constant climate",
    "predicted_sev_val": "SEV converted from prevalence",
    "adjusted_model_value_": "Post-residual, post-shift model prevalence",
}

SEV_STYLE_MAP = {
    "gbd_sev": ("black", "-"),
    "adjusted_sev_val": ("blue", "-"),
    "adjusted_sev_ssp585": ("purple", "-"),
    "adjusted_sev_ssp126": ("green", "-"),
    "predicted_sev_val": ("blue", "--"),
    "adjusted_model_value_": ("grey", ":"),
}

PREVALENCE_STYLE_MAP = {
    "gbd_value": ("black", "-"),
    "adjusted_constant_climate": ("lightgreen", "-"),
    "adjusted_ssp585": ("purple", "-"),
    "adjusted_ssp126": ("green", "-"),
    "model_value": ("blue", ":"),
    "residual_model_value": ("blue", "--"),
    "adjusted_model_value": ("blue", "-"),
}

SCENARIO_LABELS = {"ssp126": "RCP 2.6", "ssp245": "RCP 4.5", "ssp585": "RCP 8.5"}
SCENARIO_COLORS = {
    "RCP 2.6": "#046C9A",
    "RCP 4.5": "#E58601",
    "RCP 8.5": "#A42820",
}


def plot_model_comparisons_prev_print_inner(
    df: pd.DataFrame,
    ax: plt.Axes,  # type: ignore[name-defined]
    style_map: dict[str, tuple[str, str]],
) -> None:
    df = df.sort_index()
    year_index = df.index.get_level_values("year_id")

    for col, (color, linestyle) in style_map.items():
        if col in df.columns:
            label = LABEL_MAP.get(col, col)
            ax.plot(year_index, df[col], label=label, color=color, linestyle=linestyle)


def make_prevalence_plot_grid(
    cm_data: ClimateMalnutritionData,
    plot_df: pd.DataFrame,
    locs: list[int],
    out_path: str | Path,
    measure: str,
    metric: str = "prevalence",
) -> None:
    """One page per location, a sex-by-age grid of timeseries."""
    fhs_loc_meta = cm_data.load_fhs_hierarchy()
    age_meta = cm_data.load_age_group_metadata()
    style_map = SEV_STYLE_MAP if metric == "sev" else PREVALENCE_STYLE_MAP

    age_group_ids = (
        plot_df.index.get_level_values("age_group_id").unique()
        if "age_group_id" in plot_df.index.names
        else plot_df.age_group_id.unique()
    )
    sex_ids = (
        plot_df.index.get_level_values("sex_id").unique()
        if "sex_id" in plot_df.index.names
        else plot_df.sex_id.unique()
    )

    with PdfPages(out_path) as pdf:
        for loc_id in tqdm(locs):
            fig, axes = plt.subplots(
                nrows=len(sex_ids),
                ncols=len(age_group_ids),
                figsize=(14, 8),
                sharex=True,
                sharey=True,
            )

            handles, labels = [], []

            for j, sex_id in enumerate(sex_ids):  # Rows = sex
                for i, age_group_id in enumerate(age_group_ids):  # Columns = age
                    if len(sex_ids) > 1 and len(age_group_ids) > 1:
                        ax = axes[j, i]
                    elif len(age_group_ids) > 1:
                        ax = axes[i]
                    else:
                        ax = axes[j]
                    sub_df = plot_df.query(
                        "location_id == @loc_id and age_group_id == @age_group_id "
                        "and sex_id == @sex_id"
                    )
                    plot_model_comparisons_prev_print_inner(sub_df, ax, style_map)
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

                    if j == 0:
                        age_name = age_meta.query(
                            "age_group_id == @age_group_id"
                        ).age_group_name.item()
                        ax.set_title(f"Age {age_name}")

                    # Add y-axis label to leftmost column
                    if i == 0:
                        ax.set_ylabel("Male" if sex_id == 1 else "Female")
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(labelleft=False)

                    # Add x-axis label and ticks to bottom row only
                    if j == len(sex_ids) - 1:
                        ax.set_xlabel("Year")
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)

                    # Capture legend handles once
                    if not handles:
                        lines, labs = ax.get_legend_handles_labels()
                        handles, labels = lines, labs

            # Add one title for the whole figure
            title_metric = (
                measure.capitalize() + " " + "Prevalence" if "prev" in metric else "SEV"
            )
            loc_name = fhs_loc_meta.query("location_id == @loc_id").location_name.item()
            fig.suptitle(
                f"{title_metric} – Location {loc_id}: {loc_name}",
                fontsize=16,
            )

            # Add one legend at the bottom
            fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))

            fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend and title
            pdf.savefig(fig)
            plt.close(fig)


def make_results_plots(
    output_dir: Path,
    measure: str,
    results_version: str,
) -> None:
    """Make every residual step diagnostic for a results version."""
    cm_data = ClimateMalnutritionData(output_dir / measure)
    results_spec = cm_data.load_results_specification(results_version)
    scenarios = results_spec.scenarios
    fhs_loc_meta = cm_data.load_fhs_hierarchy()

    result_df = cm_data.load_sev_means(results_version)
    population = load_population_timeseries(
        None, result_df.index.get_level_values("age_group_id").unique()
    )

    result_df["adjusted_model_value"] = result_df["adjusted_model_value"].clip(0, 1)
    for scenario in scenarios:
        if scenario != REFERENCE_SCENARIO:
            result_df[f"adjusted_{scenario}"] = result_df[f"adjusted_{scenario}"].clip(
                0, 1
            )
    aggregates_forecast = aggregate_forecast_hierarchy(
        result_df, population, fhs_loc_meta, counts=False, detailed_demographics=True
    )
    locs = aggregates_forecast.index.get_level_values("location_id").unique()

    results_root = cm_data.results / results_version
    make_prevalence_plot_grid(
        cm_data,
        aggregates_forecast,
        locs,
        results_root / f"{measure}_prevalence_plots.pdf",
        measure,
        metric="prevalence",
    )
    make_prevalence_plot_grid(
        cm_data,
        aggregates_forecast,
        locs,
        results_root / f"{measure}_SEV_plots.pdf",
        measure,
        metric="sev",
    )

    plot_superregion_prevalence_rate(
        output_dir, measure, results_version, results_root / "superregion_prev.pdf"
    )
    save_cumulative_count_scenario_differences_table_measure(
        output_dir, measure, results_version
    )


def plot_superregion_prevalence_rate(
    output_dir: Path,
    measure: str,
    results_version: str,
    output_filepath: str | Path = "",
) -> None:
    plot_multiple_superregion_prevalence_rate(
        output_dir, {measure: results_version}, output_filepath
    )


def plot_multiple_superregion_prevalence_rate(  # noqa: PLR0915
    output_dir: Path,
    measures_result_version_dict: dict[str, str],
    output_filepath: str | Path = "",
) -> None:
    """Plot prevalence by scenario for the globe and each super region."""
    measures = list(measures_result_version_dict.keys())
    cm_data = ClimateMalnutritionData(output_dir / measures[0])
    hierarchy = cm_data.load_fhs_hierarchy()
    estimation_start_year = FIRST_FORECAST_YEAR

    all_prev = []
    populations = {}
    for temp_measure, temp_results_version in measures_result_version_dict.items():
        temp_cm_data = ClimateMalnutritionData(output_dir / temp_measure)
        population = load_population_timeseries(
            None, age_group_ids=VALID_AGE_GROUPS_FOR_MEASURE[temp_measure]
        )
        populations[temp_measure] = population
        temp_prev = temp_cm_data.load_shifted_prevalence(temp_results_version)
        temp_prev = temp_prev.query("year_id >= @estimation_start_year")
        temp_prev = temp_prev.clip(0, 1)
        temp_prev["measure"] = temp_measure
        temp_prev = temp_prev.set_index(["measure"], append=True)
        temp_prev = aggregate_forecast_hierarchy(
            temp_prev,
            population[temp_prev.columns],
            hierarchy,
            counts=False,
            detailed_demographics=False,
        )
        temp_prev = temp_prev.mean(axis=1).to_frame(name="prevalence")
        all_prev.append(temp_prev)
    all_prev = pd.concat(all_prev)
    modeled_locs = all_prev.index.get_level_values("location_id").unique()

    all_gbd = []
    for temp_measure in measures:
        temp_cm_data = ClimateMalnutritionData(output_dir / temp_measure)
        temp_prev = get_gbd_data(
            temp_cm_data, temp_measure, 100, draws=False, metric="prevalence"
        )
        temp_prev = temp_prev.rename(columns={"gbd_mean_prevalence": "prevalence"})
        temp_prev["scenario"] = "GBD"
        temp_prev["measure"] = temp_measure
        temp_prev = temp_prev.query("year_id >= 2000").set_index(
            ["scenario", "measure"], append=True
        )
        temp_prev = temp_prev.query("location_id in @modeled_locs")
        temp_prev = aggregate_forecast_hierarchy(
            temp_prev,
            populations[temp_measure],
            hierarchy,
            counts=False,
            detailed_demographics=False,
        )
        all_gbd.append(temp_prev)
    all_gbd = pd.concat(all_gbd)
    all_prev = pd.concat([all_prev, all_gbd])

    super_region_ids = hierarchy.query("level <= 1")["location_id"].unique()[:8]

    plot_df = all_prev.query(
        "location_id in @super_region_ids and "
        "prevalence.notna() and "
        "scenario != 'constant_climate' and "
        "measure in @measures"
    ).reset_index()

    # Merge with hierarchy and rename scenarios
    plot_df = plot_df.merge(
        hierarchy[["location_id", "location_name"]], on="location_id", how="left"
    )
    plot_df["scenario"] = plot_df["scenario"].replace(SCENARIO_LABELS)

    # Create the main figure using a constrained layout
    fig = Figure(figsize=(20, 10), layout="constrained")

    # Create a set of subfigures, stacked vertically, one per measure
    subfigs = fig.subfigures(nrows=len(measures), ncols=1)

    for plot_i, subfig in enumerate(subfigs if len(measures) > 1 else [subfigs]):
        plot_measure = measures[plot_i]

        # Create the 2x4 grid of subplots within each subfigure
        axes_block = subfig.subplots(2, 4, sharex=True, sharey=True)

        measure_df = plot_df.query("measure == @plot_measure")

        # Plot each super region in its subplot
        for i, location_id in enumerate(super_region_ids):
            row = i // 4
            col = i % 4
            ax = axes_block[row, col]

            ax.set_title(
                hierarchy.query("location_id == @location_id")["location_name"].values[
                    0
                ],
                size=20,
            )
            plot_df_sub = measure_df.query("location_id == @location_id")

            sns.lineplot(
                data=plot_df_sub,
                x="year_id",
                y="prevalence",
                hue="scenario",
                ax=ax,
                palette=SCENARIO_COLORS,
                legend=False,
                hue_order=SCENARIO_COLORS.keys(),
            )

            ax.tick_params(axis="both", which="major", labelsize=20)
            sns.despine(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")

    # Add single, centered x and y-axis labels for the entire figure
    fig.supxlabel("Year", size=22)
    if all("mort" in x for x in measures):
        fig.supylabel("Mortality Rate", size=22)
    elif any("mort" in x for x in measures):
        fig.supylabel("Prevalence / Mortality Rate", size=22)
    else:
        fig.supylabel("Prevalence", size=22)

    # Create manual legend handles
    legend_handles = [
        mlines.Line2D(
            [], [], color=color, marker=".", markersize=15, linestyle="None", label=name
        )
        for name, color in SCENARIO_COLORS.items()
    ]

    # Create the single legend at the very bottom
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(legend_handles),
        title="Representative Control Pathway (RCP) Scenario",
        title_fontsize=20,
        fontsize=20,
    )

    if output_filepath:
        fig.savefig(output_filepath, bbox_inches="tight")
    else:
        plt.show()


def save_cumulative_count_scenario_differences_table_measure(
    output_dir: Path,
    measure: str,
    results_version: str,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    loc_meta = cm_data.load_fhs_hierarchy()
    table_df = get_cumulative_count_scenario_differences_table(
        output_dir, {measure: results_version}
    )
    table_df.join(
        loc_meta.set_index("location_id")[["location_name", "sort_order"]]
    ).reset_index().sort_values(
        ["year_id", "measure", "sort_order", "scenario_order"]
    ).set_index(["year_id", "location_name", "scenario", "measure"])[
        ["formatted"]
    ].to_csv(
        cm_data.results / results_version / "table1_cumulative_case_counts.csv",
        index=True,
        encoding="utf-8-sig",
    )


def get_count_draws(
    output_dir: Path,
    measure_result_version_dict: dict[str, str],
) -> pd.DataFrame:
    """Population-weighted case count draws, aggregated up the hierarchy."""
    estimation_start_year = FIRST_FORECAST_YEAR
    counts_draws = []
    for measure, results_version in measure_result_version_dict.items():
        cm_data = ClimateMalnutritionData(output_dir / measure)
        hierarchy = cm_data.load_fhs_hierarchy()
        population = load_population_timeseries(
            None, age_group_ids=VALID_AGE_GROUPS_FOR_MEASURE[measure]
        )

        temp_prev = cm_data.load_shifted_prevalence(results_version)
        temp_prev = temp_prev.clip(0, 1)
        temp_prev = temp_prev.query("year_id >= @estimation_start_year")
        temp_prev["measure"] = measure
        temp_prev = temp_prev.set_index(["measure"], append=True)
        temp_prev = aggregate_forecast_hierarchy(
            temp_prev,
            population[temp_prev.columns],
            hierarchy,
            counts=True,
            detailed_demographics=False,
        )
        counts_draws.append(temp_prev)
    counts_draws = pd.concat(counts_draws)
    return counts_draws


def get_cumulative_count_scenario_differences_table(
    output_dir: Path,
    measure_result_version_dict: dict[str, str],
    *,
    lancet_characters: bool = False,
) -> pd.DataFrame:
    """Cumulative case count differences between pairs of scenarios."""
    estimation_start_year = FIRST_FORECAST_YEAR
    counts_draws = get_count_draws(output_dir, measure_result_version_dict)
    counts_draws = counts_draws.sort_index(
        level=["measure", "location_id", "scenario", "year_id"]
    ).query("year_id >= @estimation_start_year")

    cum_counts_draws = counts_draws.groupby(
        level=["measure", "location_id", "scenario"]
    ).cumsum()

    measures = list(measure_result_version_dict.keys())
    cm_data = ClimateMalnutritionData(output_dir / measures[0])
    loc_meta = cm_data.load_fhs_hierarchy()
    scenario_diffs = [
        ("ssp585", "ssp126"),
        ("ssp585", "ssp245"),
        ("ssp245", "ssp126"),
    ]
    analysis_years = [2050, 2100]
    table_locs = loc_meta.query("level < 2").location_id.unique()

    # Calculate the difference in cumulative counts for each scenario pair
    temp_holder = []
    for scenario_pair in scenario_diffs:
        temp_diff = cum_counts_draws.query(
            f"scenario == '{scenario_pair[0]}'"
        ).droplevel(["scenario"]) - cum_counts_draws.query(
            f"scenario == '{scenario_pair[1]}'"
        ).droplevel("scenario")
        temp_diff = temp_diff.query(
            "location_id in @table_locs and year_id in @analysis_years"
        )
        temp_mean = temp_diff.mean(axis=1).to_frame(name="mean_count")
        temp_lower_ci = temp_diff.quantile(0.025, axis=1).to_frame(name="lower_ci")
        temp_upper_ci = temp_diff.quantile(0.975, axis=1).to_frame(name="upper_ci")

        temp_reldiff = (
            cum_counts_draws.query(f"scenario == '{scenario_pair[0]}'").droplevel(
                ["scenario"]
            )
            - cum_counts_draws.query(f"scenario == '{scenario_pair[1]}'").droplevel(
                "scenario"
            )
        ).div(
            cum_counts_draws.query(f"scenario == '{scenario_pair[1]}'").droplevel(
                ["scenario"]
            )
        )
        temp_reldiff = temp_reldiff.query(
            "location_id in @table_locs and year_id in @analysis_years"
        )
        temp_reldiff_mean = temp_reldiff.mean(axis=1).to_frame(name="mean_reldiff")
        temp_reldiff_lower = temp_reldiff.quantile(0.025, axis=1).to_frame(
            name="lower_reldiff"
        )
        temp_reldiff_upper = temp_reldiff.quantile(0.975, axis=1).to_frame(
            name="upper_reldiff"
        )

        temp_info = temp_mean.join(temp_lower_ci).join(temp_upper_ci)
        temp_info = temp_info.join(temp_reldiff_mean).join(temp_reldiff_lower).join(
            temp_reldiff_upper
        )
        temp_info["scenario"] = f"{scenario_pair[0]} - {scenario_pair[1]}"
        temp_info = temp_info.reset_index().set_index(
            ["year_id", "location_id", "scenario", "measure"]
        )
        temp_holder.append(temp_info)

    table_df = pd.concat(temp_holder)

    # Presentation and formatting
    scenarios_table_sort_order = {
        "ssp585 - ssp126": 1,
        "ssp585 - ssp245": 2,
        "ssp245 - ssp126": 3,
    }
    # Single column string with mean count and CI, in millions, two decimals
    table_df["formatted"] = table_df.apply(
        lambda row: f"{row['mean_count'] / 1e6:.2f}\n"
        f"({row['lower_ci'] / 1e6:.2f}–{row['upper_ci'] / 1e6:.2f})",
        axis=1,
    )
    if lancet_characters:
        table_df["formatted"] = table_df["formatted"].str.replace(".", "·")
    table_df["scenario_order"] = table_df.index.get_level_values("scenario").map(
        scenarios_table_sort_order
    )

    return table_df
