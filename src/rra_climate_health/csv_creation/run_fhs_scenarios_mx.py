"""FHS mortality-rate (mx) scenarios from RRA's relative scenario differences.

Applies the RRA model's scenario effect to FHS's reference mortality rates (mx)
as a ratio, then turns the rates into deaths with FHS population:

    mx[s]     = fhs_mx * rra_q0[s] / rra_q0[reference]
    deaths[s] = mx[s] * population

for neonatal mortality (age group 42, 0-27 days) and under-5 mortality (age
group 1, 0-4 years). ssp245 is the reference, ssp126 the optimistic scenario
and ssp585 the pessimistic one.

Step by step, for each measure:

1. RRA q0 by scenario at RRA's most-detailed locations (draw means; 5q0 is
   rebuilt from 28q0 and 5q28 draw by draw).
2. RRA q0 aggregated to every location in the FHS hierarchy, and to both
   sexes, weighted by live births; then the ratio of each scenario to the
   reference at each location.
3. FHS's reference mx scaled by that ratio. The reference scenario is FHS's mx
   exactly.
4. Deaths = mx x population of the same age group. Population is summed from
   the most-detailed locations up to every aggregate and across sexes.

Live births are used only to aggregate RRA's q0 (step 2), since q0 is a
probability per live birth. Deaths use population (step 4), since mx is a
rate per person-year.

For each measure, writes to a new versioned directory under
``<output root>/fhs_scenarios_mx``:

- ``<measure>_mean_estimates.csv``: mx per 1,000 population and deaths in the
  Goalkeepers format of ``strun csv_creation``, 2025-2100, at the global, super
  region, region and national locations;
- ``<measure>_cumulative_scenario_differences.csv``: cumulative deaths by
  scenario and their differences, at the globe and super regions;
- ``<measure>_superregion_scenarios.png`` and ``<measure>_global_deaths.png``;

plus ``inputs.txt``, listing the results versions and files used.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FuncFormatter

from rra_climate_health import cli_options as clio
from rra_climate_health.constants import REFERENCE_SCENARIO
from rra_climate_health.csv_creation.run_csv_creation import (
    AGE_LABELS,
    BOTH_SEXES_ID,
    COUNT_INDICATORS,
    DUPLICATE_REGION_IDS,
    LIVE_BIRTHS_FILEPATH,
    LIVE_BIRTHS_MATERNAL_AGES,
    OUTPUT_COLUMNS,
    OUTPUT_YEARS,
    SCENARIO_LABELS,
    SEX_LABELS,
)
from rra_climate_health.data import (
    DEFAULT_ROOT,
    ClimateMalnutritionData,
    get_run_directory,
)

# ---- run settings -----------------------------------------------------------
# Set these to step through the script interactively, or call
# fhs_scenarios_mx_main() with no arguments to run it all with them. They are
# also the defaults of `strun fhs_scenarios_mx`, whose options override them.
OUTPUT_ROOT = Path(DEFAULT_ROOT)
NEONATAL_VERSION = "2026_10_08.01"  # neonatal_mortality results version
CHILD_MORTALITY_VERSION = "2026_10_16.01"  # child_mortality (5q28) results version

OUTPUT_DIR = "fhs_scenarios_mx"
INPUTS_FILE = "inputs.txt"

# ---- FHS inputs -------------------------------------------------------------
INPUT_ROOT = Path("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition")
# FHS reference mx, draws, every hierarchy location, sexes 1-3, age groups 42 and 1
FHS_MX_PATH = INPUT_ROOT / "fhs_mx" / "fhs_mx.nc"
# FHS population, draws, most-detailed locations (and some nationals), sexes 1-2,
# with age groups 42 and 1 added by data_prep/prep_population.py
POPULATION_PATH = INPUT_ROOT / "population_files" / "population.nc"
# Release 32 population, ages 2 + 3. Used ONLY to split national live births
# across subnational locations and sexes (step 2), never for deaths.
BIRTHS_SPLIT_POP_PATH = (
    "/mnt/share/forecasting/data/32/future/population/"
    "future_population_s130v41/population.nc"
)
BIRTHS_SPLIT_AGE_GROUPS = [2, 3]

# mx is per person-year; the CSV and plots report it per 1,000 population
PER_1000 = 1000.0

NEONATAL_AGE_GROUP_ID = 42
UNDER5_AGE_GROUP_ID = 1

# Per measure: the age group of its FHS mx, RRA q0 and population, and labels.
MEASURES = {
    "neonatal_mortality": {
        "age_group_id": NEONATAL_AGE_GROUP_ID,
        "ylabel": "Neonatal deaths per 1,000 population (0-27 days)",
        "deaths_label": "Neonatal deaths",
    },
    "child_mortality": {
        "age_group_id": UNDER5_AGE_GROUP_ID,
        "ylabel": "Under-5 deaths per 1,000 population (0-4 years)",
        "deaths_label": "Under-5 deaths",
    },
}
# CSV labels for the mx rows; the death rows use COUNT_INDICATORS
MX_INDICATORS = {
    "neonatal_mortality": ("Neonatal mortality rate (mx)", "Deaths per 1,000 population"),
    "child_mortality": ("Under-5 mortality rate (mx)", "Deaths per 1,000 population"),
}

# ---- keys, labels and analysis settings ----------------------------------
KEYS = ["location_id", "sex_id", "year_id"]
SCEN_KEYS = ["age_group_id", "sex_id", "location_id", "year_id"]
DRAW_IDX = ["location_id", "year_id", "sex_id"]
GLOBAL_LOCATION_ID = 1
# Hierarchy levels: the tables and plots cover the globe and super regions
# (levels 0-1); the CSV adds regions and countries (levels 2-3).
SUPER_REGION_LEVEL = 1
NATIONAL_LEVEL = 3
REFERENCE_SSP, BETTER_SSP, WORSE_SSP = REFERENCE_SCENARIO, "ssp126", "ssp585"
# the scenario labels used below, to the SSP each one stands for
SCENARIO_SSP = {"reference": REFERENCE_SSP, "better": BETTER_SSP, "worse": WORSE_SSP}
RRA_LEVEL_TO_SSP = {"ref": REFERENCE_SSP, "better": BETTER_SSP, "worse": WORSE_SSP}
SCENARIO_ORDER = ["ssp126", "ssp245", "ssp585"]
SCENARIO_COLOURS = ["#2a78d6", "#52514e", "#eb6834"]
# shown on the figure; the SSP names stay in the data
SCENARIO_DISPLAY = {"ssp126": "RCP 2.6", "ssp245": "RCP 4.5", "ssp585": "RCP 8.5"}
SCENARIO_LEGEND_TITLE = "Representative Control Pathways (RCP) Scenarios"
SCENARIO_PAIRS = [("ssp585", "ssp126"), ("ssp585", "ssp245"), ("ssp245", "ssp126")]
FIRST_FORECAST_YEAR = 2024
ANALYSIS_YEARS = [2050, 2100]


# =============================================================================
# Hierarchy helper
# =============================================================================


def most_detailed_ancestors(hier: pd.DataFrame) -> pd.DataFrame:
    """Each most-detailed location, paired with itself and every ancestor.

    One row per (location_id, agg_id), where agg_id runs over the location's
    own `path_to_top_parent` (the location itself included). Summing values at
    the most-detailed locations by agg_id gives the total at every location in
    the hierarchy.
    """
    anc = hier.loc[
        hier["most_detailed"] == 1, ["location_id", "path_to_top_parent"]
    ].copy()
    anc["agg_id"] = anc["path_to_top_parent"].str.split(",")
    anc = anc.explode("agg_id")
    anc["agg_id"] = anc["agg_id"].astype(int)
    return anc[["location_id", "agg_id"]]


# =============================================================================
# Step 1: RRA q0 by scenario, at RRA's most-detailed locations
# =============================================================================


def _draw_mean(path: Path, name: str) -> pd.DataFrame:
    """Mean over the draw columns of an RRA results file, keyed for a merge."""
    d = pd.read_parquet(path).mean(axis=1).rename(name).reset_index()
    return d.drop(columns=[c for c in ("scenario",) if c in d.columns])


def rra_nmr_scenario_levels(nmr_results: Path) -> pd.DataFrame:
    """The three neonatal (28q0) scenario levels at RRA's own locations."""
    paths = {s: nmr_results / f"{s}.parquet" for s in SCENARIO_ORDER}
    return (
        _draw_mean(paths[REFERENCE_SSP], "ref")
        .merge(_draw_mean(paths[BETTER_SSP], "better"), on=SCEN_KEYS)
        .merge(_draw_mean(paths[WORSE_SSP], "worse"), on=SCEN_KEYS)
    )


def rra_5q0_draws(nmr_results: Path, cm_results: Path, scenario: str) -> pd.DataFrame:
    """5q0 = 1 - (1 - 28q0) * (1 - 5q28), draw by draw, for one scenario.

    Draw by draw and not on the draw means, because the identity is not
    linear. RRA has no directly usable 5q0: the direct run falls below 28q0
    in 216 locations after 2023, so 5q0 is rebuilt from the two parts.
    """

    def _load(path: Path) -> pd.DataFrame:
        d = pd.read_parquet(path).reset_index()
        return d.drop(columns=["age_group_id", "scenario"]).set_index(DRAW_IDX)

    neo = _load(nmr_results / f"{scenario}.parquet")
    cm = _load(cm_results / f"{scenario}.parquet").reindex(neo.index)
    if cm.isna().any().any():
        msg = "28q0 and 5q28 do not cover the same location/year/sex rows"
        raise ValueError(msg)
    return 1.0 - (1.0 - neo) * (1.0 - cm)


def rra_5q0_scenario_levels(nmr_results: Path, cm_results: Path) -> pd.DataFrame:
    """The three recombined 5q0 scenario levels, at RRA's own locations."""
    draws = {
        s: rra_5q0_draws(nmr_results, cm_results, s).mean(axis=1)
        for s in SCENARIO_ORDER
    }
    d = pd.DataFrame(
        {
            "ref": draws[REFERENCE_SSP],
            "better": draws[BETTER_SSP],
            "worse": draws[WORSE_SSP],
        }
    ).reset_index()
    d["age_group_id"] = UNDER5_AGE_GROUP_ID
    return d


# =============================================================================
# Step 2: aggregate RRA q0 with live births, and form the scenario ratios
# =============================================================================


def load_births_split_population() -> pd.DataFrame:
    """Ages 2 + 3 population, by location, year and sex (births split only)."""
    return (
        xr.open_dataset(BIRTHS_SPLIT_POP_PATH)["draws"]
        .sel(age_group_id=BIRTHS_SPLIT_AGE_GROUPS)
        .sum("age_group_id")
        .mean("draw")
        .squeeze(drop=True)
        .to_dataframe(name="population")
        .reset_index()[["location_id", "year_id", "sex_id", "population"]]
    )


def load_live_births_weights(hier: pd.DataFrame) -> pd.DataFrame:
    """Live births at every RRA most-detailed location, by year and sex.

    The births file is national only and has no sex dimension, so each
    country's births are divided among its most-detailed (location, sex)
    cells in proportion to their ages 2 + 3 population:

        births[l, s] = births[country] * pop23[l, s] / pop23[country]

    National totals are therefore preserved exactly.
    """
    md = hier.loc[
        hier["most_detailed"] == 1, ["location_id", "path_to_top_parent"]
    ].copy()
    # the level-3 ancestor in each location's own path is its country
    md["country_id"] = md["path_to_top_parent"].str.split(",").str[3].astype(int)
    md = md[["location_id", "country_id"]]

    births = (
        xr.open_dataset(LIVE_BIRTHS_FILEPATH)["livebirths"]
        .sel(age_group_id=LIVE_BIRTHS_MATERNAL_AGES)
        .sum("age_group_id")
        .mean("draw")
        .squeeze(drop=True)
        .to_dataframe(name="country_births")
        .reset_index()[["location_id", "year_id", "country_births"]]
        .rename(columns={"location_id": "country_id"})
    )

    pop = load_births_split_population().merge(md, on="location_id", how="inner")
    pop["country_pop"] = pop.groupby(["country_id", "year_id"])["population"].transform(
        "sum"
    )
    w = pop.merge(births, on=["country_id", "year_id"], how="inner")
    w["births"] = w["country_births"] * w["population"] / w["country_pop"]
    return w[["location_id", "year_id", "sex_id", "births"]]


def aggregate_scenarios(
    scenarios: pd.DataFrame,
    weights: pd.DataFrame,
    age_group_id: int,
    hier: pd.DataFrame,
) -> pd.DataFrame:
    """RRA's scenario levels and ratios at every location in the hierarchy.

    Each most-detailed location lands in every ancestor, and each scenario's
    q0 is averaged there weighted by live births. sex_id 3 is added as the
    both-sex aggregate so FHS's both-sexes rows can be matched. Summed births
    travel with it, for RRA's own death counts (rra_modeled_counts).

    The ratios are formed AFTER aggregating: at each location, the ratio of
    the births-weighted scenario level to the births-weighted reference, which
    is the same as the ratio of RRA's death counts there. Unlike a difference,
    a ratio is not linear, so averaging the most-detailed ratios instead would
    give a different answer.
    """
    d = scenarios.merge(weights, on=DRAW_IDX, how="inner")
    d = d.merge(most_detailed_ancestors(hier), on="location_id")

    levels = ["ref", "better", "worse"]
    for col in levels:
        d[f"_w_{col}"] = d[col] * d["births"]

    def _collapse(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        out = frame.groupby(keys, as_index=False)[
            [f"_w_{c}" for c in levels] + ["births"]
        ].sum()
        for col in levels:
            out[col] = out[f"_w_{col}"] / out["births"]
        return out.drop(columns=[f"_w_{c}" for c in levels])

    by_sex = _collapse(d, ["agg_id", "year_id", "sex_id"])
    both = _collapse(d, ["agg_id", "year_id"])
    both["sex_id"] = BOTH_SEXES_ID
    out = pd.concat([by_sex, both], ignore_index=True)
    if (out["ref"] <= 0).any():
        msg = "RRA reference is zero or negative somewhere; the ratio is undefined"
        raise ValueError(msg)
    out["ratio_better"] = out["better"] / out["ref"]
    out["ratio_worse"] = out["worse"] / out["ref"]
    out["age_group_id"] = age_group_id
    return out.rename(columns={"agg_id": "location_id"})


# =============================================================================
# Step 3: scale FHS's reference mx by the RRA ratio
# =============================================================================


def load_fhs_mx(age_group_id: int) -> pd.DataFrame:
    """FHS reference mx for one age group: draw mean, by location, sex and year."""
    return (
        xr.open_dataset(FHS_MX_PATH)["draws"]
        .sel(age_group_id=age_group_id, drop=True)
        .mean("draw")
        .to_dataframe(name="fhs_mx")
        .reset_index()[[*KEYS, "fhs_mx"]]
    )


def apply_ratios_to_fhs_mx(
    fhs_mx: pd.DataFrame, ratios: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """FHS reference mx, scaled by the aggregated RRA scenario ratio.

        mx[reference] = fhs_mx
        mx[better]    = fhs_mx * ratio_better
        mx[worse]     = fhs_mx * ratio_worse

    The ratio is RRA's relative change in q0, used here as the relative change
    in mx. For small probabilities of death the two are close. Only locations
    in both FHS's mx and the hierarchy are kept. The floor at zero is the last
    thing applied; a positive rate times a positive ratio cannot go negative,
    so it is there as a guard and its count is returned.

    Returns one row per location, sex, year and scenario, with column ``mx``.
    """
    wide = fhs_mx.merge(
        ratios[[*KEYS, "ratio_better", "ratio_worse"]], on=KEYS, how="inner"
    )
    scenario_ratio = {
        "reference": 1.0,
        "better": wide["ratio_better"],
        "worse": wide["ratio_worse"],
    }
    out = pd.concat(
        [
            wide[KEYS].assign(scenario=lab, mx=wide["fhs_mx"] * ratio)
            for lab, ratio in scenario_ratio.items()
        ],
        ignore_index=True,
    )

    clipped = {
        lab: int(((out["scenario"] == lab) & (out["mx"] < 0)).sum())
        for lab in scenario_ratio
    }
    out["mx"] = out["mx"].clip(lower=0.0)
    return out, clipped


# =============================================================================
# Step 4: deaths = mx x population of the same age group
# =============================================================================


def load_age_group_population(age_group_id: int, hier: pd.DataFrame) -> pd.DataFrame:
    """Population of one age group at every hierarchy location, by sex and both.

    The file has the most-detailed locations (plus some nationals) and males
    and females only, so population is summed from the most-detailed
    locations to every location that contains them, and across sexes:

        population[A, sex]  = sum over most-detailed l in A of population[l, sex]
        population[A, both] = population[A, male] + population[A, female]

    This is the same aggregation as the live births in step 2.
    """
    pop = (
        xr.open_dataset(POPULATION_PATH)["draws"]
        .sel(age_group_id=age_group_id, drop=True)
        .mean("draw")
        .to_dataframe(name="population")
        .reset_index()[[*KEYS, "population"]]
    )
    # the inner merge keeps the most-detailed locations only
    pop = pop.merge(most_detailed_ancestors(hier), on="location_id", how="inner")

    by_sex = pop.groupby(["agg_id", "sex_id", "year_id"], as_index=False)[
        "population"
    ].sum()
    both = by_sex.groupby(["agg_id", "year_id"], as_index=False)["population"].sum()
    both["sex_id"] = BOTH_SEXES_ID
    return pd.concat([by_sex, both], ignore_index=True).rename(
        columns={"agg_id": "location_id"}
    )


def add_deaths(scenarios: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Population and deaths alongside each scenario's mx.

        deaths = mx * population

    Every scenario row must have a population: no row is dropped silently.
    """
    out = scenarios.merge(population, on=KEYS, how="left")
    if out["population"].isna().any():
        missing = out.loc[out["population"].isna(), "location_id"].unique()
        msg = f"No population for location_ids {sorted(missing)}"
        raise ValueError(msg)
    out["deaths"] = out["mx"] * out["population"]
    return out


# =============================================================================
# Outputs: CSV, cumulative table and plots
# =============================================================================


def table_series(scen: pd.DataFrame, hier: pd.DataFrame) -> pd.DataFrame:
    """Deaths and mx per 1,000 at the globe and super regions, both sexes.

    The series behind the cumulative table and the plots.
    """
    table_locs = set(hier.loc[hier["level"] <= SUPER_REGION_LEVEL, "location_id"])
    out = scen[
        scen["location_id"].isin(table_locs) & (scen["sex_id"] == BOTH_SEXES_ID)
    ].copy()
    out["count"] = out["deaths"]
    out["mx_per_1000"] = out["mx"] * PER_1000
    out["scenario"] = out["scenario"].map(SCENARIO_SSP)
    return out[
        ["location_id", "scenario", "year_id", "count", "population", "mx_per_1000"]
    ].reset_index(drop=True)


def goalkeepers_csv(
    scenarios: pd.DataFrame, measure: str, hier: pd.DataFrame
) -> pd.DataFrame:
    """The scenario series in the Goalkeepers results format.

    One row per metric, location, sex, scenario and year (OUTPUT_YEARS), for
    the global, super region, region and national locations, labelled as in
    ``strun csv_creation``. Two metrics: mx per 1,000 population, and deaths
    (mx times the population of the same age group, as in add_deaths).
    """
    locations = hier.loc[
        (hier["level"] <= NATIONAL_LEVEL)
        & ~hier["location_id"].isin(DUPLICATE_REGION_IDS),
        ["location_id", "super_region_name", "region_name", "location_ascii_name"],
    ]
    mx_indicator, mx_units = MX_INDICATORS[measure]
    count_indicator, count_units = COUNT_INDICATORS[measure]

    df = scenarios[scenarios["year_id"].isin(OUTPUT_YEARS)].merge(
        locations, on="location_id", how="inner"
    )
    missing = set(locations["location_id"]) - set(df["location_id"])
    if missing:
        msg = f"No scenario series for location_ids {sorted(missing)}"
        raise ValueError(msg)
    missing_years = set(OUTPUT_YEARS) - set(df["year_id"])
    if missing_years:
        msg = f"No scenario series for years {sorted(missing_years)}"
        raise ValueError(msg)
    df = df.assign(
        **{
            "Sex": df["sex_id"].map(SEX_LABELS),
            "Scenario type": df["scenario"].map(SCENARIO_SSP).map(SCENARIO_LABELS),
            "Age": AGE_LABELS[measure],
        }
    )
    rates = df.assign(
        **{
            "Indicator": mx_indicator,
            "Mean estimate": df["mx"] * PER_1000,
            "Indicator units": mx_units,
        }
    )
    counts = df.assign(
        **{
            "Indicator": count_indicator,
            "Mean estimate": df["deaths"],
            "Indicator units": count_units,
        }
    )
    df = pd.concat([rates, counts], ignore_index=True).rename(
        columns={
            "super_region_name": "Super region",
            "region_name": "Region",
            "location_ascii_name": "Location",
            "year_id": "Year",
        }
    )
    if df[["Sex", "Scenario type", "Mean estimate"]].isna().any().any():
        msg = "unmapped sex_id or scenario label, or no deaths for a row"
        raise ValueError(msg)
    # rates before counts
    df["_count"] = df["Indicator"] == count_indicator
    return df.sort_values(["_count", "location_id", "Sex", "Scenario type", "Year"])[
        OUTPUT_COLUMNS
    ]


def rra_modeled_counts(agg: pd.DataFrame) -> pd.DataFrame:
    """RRA's own deaths at Global, both sexes, before any FHS step.

        deaths = births-weighted RRA q0 x live births

    q0 is a probability per live birth, so RRA's own deaths use live births.
    Same location, scenario, year and count columns as table_series.
    """
    g = agg[
        (agg["location_id"] == GLOBAL_LOCATION_ID) & (agg["sex_id"] == BOTH_SEXES_ID)
    ]
    out = g.melt(
        id_vars=["location_id", "year_id", "births"],
        value_vars=list(RRA_LEVEL_TO_SSP),
        var_name="level",
        value_name="prob",
    )
    out["scenario"] = out["level"].map(RRA_LEVEL_TO_SSP)
    out["count"] = out["prob"] * out["births"]
    return out[["location_id", "scenario", "year_id", "count"]].reset_index(drop=True)


def cumulative_difference_table(
    counts: pd.DataFrame, hier: pd.DataFrame
) -> pd.DataFrame:
    """Cumulative deaths by scenario, the differences, and each as a percent.

    At ANALYSIS_YEARS, in whole deaths. For each location the three
    scenarios' own cumulative totals come first, then the three differences:

        cumulative[s]   = sum over 2024..year of deaths[s]
        deaths          = cumulative[A] - cumulative[B]
        pct_change_from_lower_scenario
                        = 100 * (cumulative[A] - cumulative[B]) / cumulative[B]

    B is always the lower-emission scenario of the pair. The reference row is
    FHS's own series; the other two are FHS scaled by the RRA scenario ratio.
    """
    cum = counts.query("year_id >= @FIRST_FORECAST_YEAR").sort_values("year_id").copy()
    cum["cum_count"] = cum.groupby(["location_id", "scenario"])["count"].cumsum()
    wide = cum.pivot_table(
        index=["location_id", "year_id"], columns="scenario", values="cum_count"
    )
    rows = []
    for s in SCENARIO_ORDER:
        d = wide[s].rename("value").reset_index()
        d["pct"] = np.nan
        d["scenario"] = s
        rows.append(d)
    for a, b in SCENARIO_PAIRS:
        d = (wide[a] - wide[b]).rename("value").reset_index()
        d["pct"] = (100 * (wide[a] - wide[b]) / wide[b]).to_numpy()
        d["scenario"] = f"{a} - {b}"
        rows.append(d)
    row_order = {
        lab: k
        for k, lab in enumerate(
            SCENARIO_ORDER + [f"{a} - {b}" for a, b in SCENARIO_PAIRS]
        )
    }
    table = (
        pd.concat(rows)
        .query("year_id in @ANALYSIS_YEARS")
        .merge(
            hier[["location_id", "location_name", "sort_order"]],
            on="location_id",
            how="left",
        )
    )
    table["_row"] = table["scenario"].map(row_order)
    table = table.sort_values(["year_id", "sort_order", "_row"])
    table["deaths"] = table["value"].round().astype("int64")
    table["pct_change_from_lower_scenario"] = table["pct"].round(2)
    return table[
        [
            "year_id",
            "location_name",
            "scenario",
            "deaths",
            "pct_change_from_lower_scenario",
        ]
    ].reset_index(drop=True)


def plot_superregion_rates(
    counts: pd.DataFrame, ylabel: str, hier: pd.DataFrame, ncol: int = 4
) -> plt.Figure:
    """mx per 1,000 by scenario, one panel per globe/super region, shared y axis."""
    locs = hier[
        hier["location_id"].isin(counts.location_id.unique())
        & (hier["level"] <= SUPER_REGION_LEVEL)
    ].sort_values("sort_order")
    nrow = int(np.ceil(len(locs) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), sharex=True, sharey=True
    )
    handles = []
    for ax, (_, loc) in zip(axes.flatten(), locs.iterrows(), strict=False):
        panel = counts[counts["location_id"] == loc["location_id"]]
        for scen, colour in zip(SCENARIO_ORDER, SCENARIO_COLOURS, strict=True):
            s = panel[panel["scenario"] == scen].sort_values("year_id")
            (line,) = ax.plot(s["year_id"], s["mx_per_1000"], color=colour, lw=2)
            if len(handles) < len(SCENARIO_ORDER):
                line.set_label(SCENARIO_DISPLAY[scen])
                handles.append(line)
        ax.set_title(loc["location_name"], fontsize=12)
        ax.grid(visible=True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
    for ax in axes.flatten()[len(locs) :]:
        ax.set_visible(False)
    fig.supylabel(ylabel, fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(
        handles=handles,
        title=SCENARIO_LEGEND_TITLE,
        loc="lower center",
        ncol=len(SCENARIO_ORDER),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
        bbox_to_anchor=(0.5, 0.0),
    )
    return fig


def _global_cumulative(counts: pd.DataFrame) -> pd.DataFrame:
    """Global rows from FIRST_FORECAST_YEAR, with deaths cumulated by scenario."""
    g = (
        counts[
            (counts["location_id"] == GLOBAL_LOCATION_ID)
            & (counts["year_id"] >= FIRST_FORECAST_YEAR)
        ]
        .sort_values("year_id")
        .copy()
    )
    g["cum_count"] = g.groupby("scenario")["count"].cumsum()
    return g


def _annotate_gap(
    ax: plt.Axes, frame: pd.DataFrame, col: str, year: int, label: str
) -> None:
    """Bracket RCP 8.5 - RCP 2.6 at `year`, labelled in deaths and percent.

    The label sits left of the bracket, except at the earlier analysis years
    on the cumulative panel: there the counts run to eight digits and a
    left-hand label runs off the start of the axis, while the cumulative
    lines rise away to the right and leave room for it.
    """
    on_left = col != "cum_count" or year == max(ANALYSIS_YEARS)
    at = frame[frame["year_id"] == year].set_index("scenario")[col]
    lo, hi = at[BETTER_SSP], at[WORSE_SSP]
    ax.annotate(
        "",
        xy=(year, lo),
        xytext=(year, hi),
        # flat caps: arrowheads collide into an X when the gap is small
        arrowprops={
            "arrowstyle": "|-|,widthA=0.3,widthB=0.3",
            "color": "black",
            "lw": 1,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.annotate(
        f"{label}: {hi - lo:,.0f} ({100 * (hi - lo) / lo:+.1f}%)",
        xy=(year, (lo + hi) / 2),
        xytext=(-6 if on_left else 6, 0),
        textcoords="offset points",
        ha="right" if on_left else "left",
        va="center",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )


def _series_label(source: str, scen: str) -> str:
    """Legend label for one line: FHS (solid) or RRA model (dashed)."""
    if source == "FHS":
        ref = " (reference)" if scen == REFERENCE_SSP else ""
        return f"FHS, {SCENARIO_DISPLAY[scen]}{ref}"
    return f"RRA model, {SCENARIO_DISPLAY[scen]}"


def plot_global_deaths(
    fhs_counts: pd.DataFrame, rra_counts: pd.DataFrame, deaths_label: str
) -> plt.Figure:
    """Global deaths by scenario: per year (left) and cumulative (right).

    Solid: FHS reference and FHS scaled by the RRA ratio - the series in the
    tables, so the right panel at 2050 and 2100 reads off the table's Global
    level rows. Dashed: RRA's own deaths for all three scenarios, RCP 4.5
    included as the ratio's denominator. Brackets give RCP 8.5 - RCP 2.6 at
    ANALYSIS_YEARS, in deaths and as a percent of RCP 2.6.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharex=True)
    fhs = _global_cumulative(fhs_counts)
    rra = _global_cumulative(rra_counts)
    colour = dict(zip(SCENARIO_ORDER, SCENARIO_COLOURS, strict=True))
    series = [(fhs, scen, "-", _series_label("FHS", scen)) for scen in SCENARIO_ORDER]
    series += [(rra, scen, "--", _series_label("RRA", scen)) for scen in SCENARIO_ORDER]
    whole = FuncFormatter(lambda v, _: f"{v:,.0f}")

    handles = {}
    for ax, col, title in zip(
        axes,
        ["count", "cum_count"],
        [
            f"{deaths_label} per year",
            f"Cumulative {deaths_label.lower()} since {FIRST_FORECAST_YEAR}",
        ],
        strict=True,
    ):
        for frame, scen, style, label in series:
            s = frame[frame["scenario"] == scen]
            (line,) = ax.plot(
                s["year_id"], s[col], color=colour[scen], lw=2, linestyle=style
            )
            handles.setdefault(label, line)
        for year in ANALYSIS_YEARS:
            _annotate_gap(ax, rra, col, year, "RRA 8.5 - 2.6")
            _annotate_gap(ax, fhs, col, year, "FHS 8.5 - 2.6")
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(whole)
        ax.grid(visible=True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    order = [
        _series_label(src, scen) for src in ("FHS", "RRA") for scen in SCENARIO_ORDER
    ]
    labels = [lab for lab in order if lab in handles]
    fig.legend(
        handles=[handles[lab] for lab in labels],
        labels=labels,
        title=SCENARIO_LEGEND_TITLE,
        ncol=len(labels),
        frameon=False,
        fontsize=10,
        title_fontsize=12,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    return fig


# =============================================================================
# Run
# =============================================================================


def run_measure(
    measure: str,
    levels: pd.DataFrame,
    births: pd.DataFrame,
    hier: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Steps 2-4 for one measure, then its CSV, table and plots."""
    age_group_id = MEASURES[measure]["age_group_id"]

    # step 2: RRA q0 aggregated with live births; scenario ratios
    agg = aggregate_scenarios(levels, births, age_group_id, hier)
    # step 3: FHS reference mx scaled by the ratios
    scenarios, clipped = apply_ratios_to_fhs_mx(load_fhs_mx(age_group_id), agg)
    # step 4: deaths = mx x population of the same age group
    scenarios = add_deaths(scenarios, load_age_group_population(age_group_id, hier))

    series = table_series(scenarios, hier)
    table = cumulative_difference_table(series, hier)

    csv_path = run_dir / f"{measure}_mean_estimates.csv"
    table_path = run_dir / f"{measure}_cumulative_scenario_differences.csv"
    plot_path = run_dir / f"{measure}_superregion_scenarios.png"
    global_path = run_dir / f"{measure}_global_deaths.png"

    goalkeepers_csv(scenarios, measure, hier).to_csv(csv_path, index=False)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    fig = plot_superregion_rates(series, MEASURES[measure]["ylabel"], hier)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig = plot_global_deaths(
        series, rra_modeled_counts(agg), MEASURES[measure]["deaths_label"]
    )
    fig.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n== {measure}: rows floored at zero {clipped}")
    for path in (csv_path, table_path, plot_path, global_path):
        print(f"  wrote {path}")
    print(table.query("location_name == 'Global'").to_string(index=False))


def fhs_scenarios_mx_main(
    output_root: Path = OUTPUT_ROOT,
    neonatal_version: str = NEONATAL_VERSION,
    child_mortality_version: str = CHILD_MORTALITY_VERSION,
) -> Path:
    """Run steps 1-4 for both measures and write every output to a new run_dir."""
    nmr_results = (
        ClimateMalnutritionData(output_root / "neonatal_mortality").results
        / neonatal_version
    )
    cm_data = ClimateMalnutritionData(output_root / "child_mortality")
    cm_results = cm_data.results / child_mortality_version
    for results in (nmr_results, cm_results):
        if missing := [
            s for s in SCENARIO_ORDER if not (results / f"{s}.parquet").exists()
        ]:
            message = (
                f"{results} has no inference draws for {missing}. Run inference "
                "for this results version first."
            )
            raise click.ClickException(message)

    hier = cm_data.load_fhs_hierarchy()
    out_root = output_root / OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = get_run_directory(out_root)
    run_dir.mkdir()
    (run_dir / INPUTS_FILE).write_text(
        f"neonatal_mortality results: {nmr_results}\n"
        f"child_mortality results:    {cm_results}\n"
        f"FHS mx:                     {FHS_MX_PATH}\n"
        f"population (deaths):        {POPULATION_PATH}\n"
        f"live births (RRA weights):  {LIVE_BIRTHS_FILEPATH}\n"
        f"population (births split):  {BIRTHS_SPLIT_POP_PATH}\n"
    )

    # step 1: RRA q0 by scenario; births for step 2
    births = load_live_births_weights(hier)
    run_measure(
        "neonatal_mortality",
        rra_nmr_scenario_levels(nmr_results),
        births,
        hier,
        run_dir,
    )
    run_measure(
        "child_mortality",
        rra_5q0_scenario_levels(nmr_results, cm_results),
        births,
        hier,
        run_dir,
    )
    return run_dir


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(OUTPUT_ROOT)
@click.option(
    "--neonatal-version",
    default=NEONATAL_VERSION,
    show_default=True,
    help="neonatal_mortality results version whose inference draws to use.",
)
@click.option(
    "--child-mortality-version",
    default=CHILD_MORTALITY_VERSION,
    show_default=True,
    help=(
        "child_mortality (5q28) results version whose inference draws to use; "
        "combined with the neonatal draws into 5q0."
    ),
)
def run_fhs_scenarios_mx(
    output_root: str,
    neonatal_version: str,
    child_mortality_version: str,
) -> None:
    """Apply RRA's relative scenario differences to FHS neonatal and under-5 mx."""
    run_dir = fhs_scenarios_mx_main(
        Path(output_root), neonatal_version, child_mortality_version
    )
    print(f"\nFHS mx scenario outputs written to {run_dir}")
