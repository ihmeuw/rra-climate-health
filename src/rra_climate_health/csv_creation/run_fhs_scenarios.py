"""FHS mortality scenarios from RRA's relative scenario differences.

Applies the RRA model's scenario effect to the published FHS mortality
forecasts as a ratio:

    out = fhs_published * rra_scenario / rra_reference

for neonatal mortality (28q0) and under-5 mortality (5q0). ssp245 is the
reference, ssp126 the optimistic scenario and ssp585 the pessimistic one. RRA's
gap between scenarios carries over to FHS in PERCENT, and FHS's own level sets
its size in deaths. RRA comes from the original model draws and is the only
thing aggregated (live-births weighted); FHS is used exactly as published, so
the reference scenario is FHS. The floor at zero is the last thing applied.

For each measure, writes to a new versioned directory under
``<output root>/fhs_scenarios``:

- ``<measure>_mean_estimates.csv``: the scenario series in the Goalkeepers
  format of ``strun csv_creation`` -- rates and death counts, 2025-2100, at the
  global, super region, region and national locations;
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
    INDICATORS,
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

OUTPUT_DIR = "fhs_scenarios"
INPUTS_FILE = "inputs.txt"

# ---- FHS -------------------------------------------------------------------
# Release 32, scenario 130: the same release as the live births and the
# population used to split them.
FHS_Q0_ROOT = Path(
    "/mnt/share/forecasting/data/32/future/death/"
    "future_death_aggregated_rate_s130v156_q0_indicators"
)
# FHS publishes deaths per 1000 livebirths; RRA works in probability
FHS_RATE_PER = 1000.0
# Population, ages 2 + 3. Used ONLY to split national births across
# subnational locations and sexes, never as a weight in its own right.
FHS_POP_PATH = (
    "/mnt/share/forecasting/data/32/future/population/"
    "future_population_s130v41/population.nc"
)
NEONATAL_POP_PARTS = [2, 3]

NEONATAL_AGE_GROUP_ID = 42
UNDER5_AGE_GROUP_ID = 1

# Per measure: the age group of its FHS and RRA series, its FHS file, and the
# plot labels.
MEASURES = {
    "neonatal_mortality": {
        "age_group_id": NEONATAL_AGE_GROUP_ID,
        "fhs_file": "nmr_28dayq0.csv",
        "ylabel": "Neonatal deaths per 1000 livebirths",
        "deaths_label": "Neonatal deaths",
    },
    "child_mortality": {
        "age_group_id": UNDER5_AGE_GROUP_ID,
        "fhs_file": "u5mr_5q0.csv",
        "ylabel": "Under-5 deaths per 1000 livebirths",
        "deaths_label": "Under-5 deaths",
    },
}

# ---- keys, labels and analysis settings ----------------------------------
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


def _draw_mean(path: Path, name: str) -> pd.DataFrame:
    """Mean over the draw columns of an RRA results file, keyed for a merge."""
    d = pd.read_parquet(path).mean(axis=1).rename(name).reset_index()
    return d.drop(columns=[c for c in ("scenario",) if c in d.columns])


def load_neonatal_population() -> pd.DataFrame:
    """Ages 2 + 3 population, by location, year and sex."""
    return (
        xr.open_dataset(FHS_POP_PATH)["draws"]
        .sel(age_group_id=NEONATAL_POP_PARTS)
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

    pop = load_neonatal_population().merge(md, on="location_id", how="inner")
    pop["country_pop"] = pop.groupby(["country_id", "year_id"])["population"].transform(
        "sum"
    )
    w = pop.merge(births, on=["country_id", "year_id"], how="inner")
    w["births"] = w["country_births"] * w["population"] / w["country_pop"]
    return w[["location_id", "year_id", "sex_id", "births"]]


def rra_nmr_scenario_levels(nmr_results: Path) -> pd.DataFrame:
    """The three neonatal scenario levels at RRA's own locations."""
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


def aggregate_scenarios(
    scenarios: pd.DataFrame,
    weights: pd.DataFrame,
    age_group_id: int,
    hier: pd.DataFrame,
) -> pd.DataFrame:
    """RRA's scenario levels and ratios at every location in the hierarchy.

    Explode `path_to_top_parent` so each most-detailed location lands in every
    ancestor, then take the births-weighted mean of each scenario. sex_id 3 is
    added as the both-sex aggregate so FHS's "Both" rows can be matched.
    Summed births travel with it, for the counts step.

    The ratios are formed AFTER aggregating: at each location, the ratio of
    the births-weighted scenario level to the births-weighted reference, which
    is the same as the ratio of RRA's death counts there. Unlike a difference,
    a ratio is not linear, so averaging the most-detailed ratios instead would
    give a different answer.
    """
    d = scenarios.merge(weights, on=DRAW_IDX, how="inner")
    anc = hier.loc[
        hier["most_detailed"] == 1, ["location_id", "path_to_top_parent"]
    ].copy()
    anc["agg_id"] = anc["path_to_top_parent"].str.split(",")
    anc = anc.explode("agg_id")
    anc["agg_id"] = anc["agg_id"].astype(int)
    d = d.merge(anc[["location_id", "agg_id"]], on="location_id")

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


def apply_ratios_to_fhs(
    fhs_path: Path, ratios: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """FHS series as published, scaled by the aggregated RRA scenario ratio.

        out = fhs_published * rra_scenario / rra_reference

    The reference is FHS exactly. Only the central estimate is carried
    through; FHS upper/lower are dropped rather than scaled, since a scaled
    interval would look like a modelled one. The floor at zero is the last
    thing applied; a positive rate times a positive ratio cannot go negative,
    so it is there as a guard and its count is returned.
    """
    fhs = pd.read_csv(fhs_path)
    wide = fhs.merge(
        ratios[SCEN_KEYS + ["ratio_better", "ratio_worse", "births"]],
        on=SCEN_KEYS,
        how="inner",
    )
    keep = [*fhs.columns.drop(["upper", "lower"]), "births"]
    base = wide[keep]
    parts = [base.assign(scenario="reference")]
    for lab in ("better", "worse"):
        p = base.copy()
        p["scenario"] = lab
        p["mean"] = (wide["mean"] * wide[f"ratio_{lab}"]).to_numpy()
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)

    clipped = {
        lab: int(((out["scenario"] == lab) & (out["mean"] < 0)).sum())
        for lab in ("reference", "better", "worse")
    }
    out["mean"] = out["mean"].clip(lower=0.0)
    return out, clipped


def counts_from_scenarios(scen: pd.DataFrame, hier: pd.DataFrame) -> pd.DataFrame:
    """Deaths and live births, at the globe and super regions, Both sexes.

    deaths = rate per 1000 / 1000 * live births. The rates come from FHS at
    each aggregate level, and the births are the most-detailed births summed
    to that same level, so nothing is summed up the hierarchy here.
    """
    table_locs = set(hier.loc[hier["level"] <= SUPER_REGION_LEVEL, "location_id"])
    out = scen[
        scen["location_id"].isin(table_locs) & (scen["sex_id"] == BOTH_SEXES_ID)
    ].copy()
    out["count"] = out["mean"] / FHS_RATE_PER * out["births"]
    out["rate"] = out["mean"]
    out["scenario"] = out["scenario"].map(SCENARIO_SSP)
    return out[
        ["location_id", "scenario", "year_id", "count", "births", "rate"]
    ].reset_index(drop=True)


def goalkeepers_csv(
    scenarios: pd.DataFrame, measure: str, hier: pd.DataFrame
) -> pd.DataFrame:
    """The scenario series in the Goalkeepers results format.

    One row per metric, location, sex, scenario and year (OUTPUT_YEARS), for
    the global, super region, region and national locations, labelled as in
    ``strun csv_creation``. FHS publishes all of them, both sexes included, so
    the rate is the FHS rate per 1000 as it stands; the count is that rate
    times the live births summed to the same location, as in
    counts_from_scenarios.
    """
    locations = hier.loc[
        (hier["level"] <= NATIONAL_LEVEL)
        & ~hier["location_id"].isin(DUPLICATE_REGION_IDS),
        ["location_id", "super_region_name", "region_name", "location_ascii_name"],
    ]
    indicator, units, _ = INDICATORS[measure]
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
            "Indicator": indicator,
            "Mean estimate": df["mean"],
            "Indicator units": units,
        }
    )
    counts = df.assign(
        **{
            "Indicator": count_indicator,
            "Mean estimate": df["mean"] / FHS_RATE_PER * df["births"],
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
        msg = "unmapped sex_id or scenario label, or no live births for a count"
        raise ValueError(msg)
    # rates before counts
    df["_count"] = df["Indicator"] == count_indicator
    return df.sort_values(["_count", "location_id", "Sex", "Scenario type", "Year"])[
        OUTPUT_COLUMNS
    ]


def rra_modeled_counts(agg: pd.DataFrame) -> pd.DataFrame:
    """RRA's own deaths at Global, Both sexes, before any FHS step.

        deaths = births-weighted RRA probability x live births

    Same columns as counts_from_scenarios, rate per 1000.
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
    out["rate"] = out["prob"] * FHS_RATE_PER
    return out[
        ["location_id", "scenario", "year_id", "count", "births", "rate"]
    ].reset_index(drop=True)


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
    """Rate by scenario, one panel per globe/super region, shared y axis."""
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
            (line,) = ax.plot(s["year_id"], s["rate"], color=colour, lw=2)
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
    level rows. Dashed: RRA's own levels for all three scenarios, RCP 4.5
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


def run_measure(
    measure: str,
    levels: pd.DataFrame,
    births: pd.DataFrame,
    hier: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Build, tabulate, plot and export one measure with the relative method."""
    spec = MEASURES[measure]
    agg = aggregate_scenarios(levels, births, spec["age_group_id"], hier)
    scenarios, clipped = apply_ratios_to_fhs(FHS_Q0_ROOT / spec["fhs_file"], agg)
    counts = counts_from_scenarios(scenarios, hier)
    table = cumulative_difference_table(counts, hier)

    csv_path = run_dir / f"{measure}_mean_estimates.csv"
    table_path = run_dir / f"{measure}_cumulative_scenario_differences.csv"
    plot_path = run_dir / f"{measure}_superregion_scenarios.png"
    global_path = run_dir / f"{measure}_global_deaths.png"

    goalkeepers_csv(scenarios, measure, hier).to_csv(csv_path, index=False)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    fig = plot_superregion_rates(counts, spec["ylabel"], hier)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig = plot_global_deaths(counts, rra_modeled_counts(agg), spec["deaths_label"])
    fig.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n== {measure}: rows floored at zero {clipped}")
    for path in (csv_path, table_path, plot_path, global_path):
        print(f"  wrote {path}")
    print(table.query("location_name == 'Global'").to_string(index=False))


def fhs_scenarios_main(
    output_root: Path,
    neonatal_version: str,
    child_mortality_version: str,
) -> Path:
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
        f"FHS q0 indicators:          {FHS_Q0_ROOT}\n"
        f"live births:                {LIVE_BIRTHS_FILEPATH}\n"
        f"population (births split):  {FHS_POP_PATH}\n"
    )

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
@clio.with_output_root(DEFAULT_ROOT)
@click.option(
    "--neonatal-version",
    required=True,
    help="neonatal_mortality results version whose inference draws to use.",
)
@click.option(
    "--child-mortality-version",
    required=True,
    help=(
        "child_mortality (5q28) results version whose inference draws to use; "
        "combined with the neonatal draws into 5q0."
    ),
)
def run_fhs_scenarios(
    output_root: str,
    neonatal_version: str,
    child_mortality_version: str,
) -> None:
    """Apply RRA's relative scenario differences to FHS neonatal and under-5 mortality."""
    run_dir = fhs_scenarios_main(
        Path(output_root), neonatal_version, child_mortality_version
    )
    print(f"\nFHS scenario outputs written to {run_dir}")
