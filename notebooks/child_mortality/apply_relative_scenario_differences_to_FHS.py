"""Apply RRA's RELATIVE scenario differences to the FHS mortality series.

Script version of apply_absolute_scenario_differences_to_FHS.ipynb, with the
RRA scenario effect applied as a ratio rather than a difference:

    absolute (the notebook)   out = fhs_published + (rra_scenario - rra_reference) * 1000
    relative (this script)    out = fhs_published * rra_scenario / rra_reference

ssp245 is the reference, ssp126 the better scenario and ssp585 the worse. RRA's
gap between scenarios carries over to FHS in PERCENT, and FHS's own level sets
its size in deaths. Everything else follows the notebook: RRA comes from the
original model draws, only RRA is aggregated (live-births weighted), FHS is
used exactly as published, and the floor at zero is the last thing applied.

Besides the plots and cumulative tables, each indicator's scenario series is
written as a CSV in the Goalkeepers format that `strun csv_creation` produces
in the rra-climate-health_2 clone. The FHS series already holds every location
and both-sex row, so the CSV rates are a relabelling of it, never a
re-aggregation; death counts are those rates times live births. 2025-2100 only.

Every file this script writes has OUTPUT_LABEL in its name, so none of the
absolute-differences outputs - or the notebook's *_global_deaths_relative.png
comparison plots - are overwritten.

Run with the gbd_env python:

    /ihme/code/central_comp/miniconda/envs/gbd_env/bin/python \
        apply_relative_scenario_differences_to_FHS.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PLOT_OUTPUT = (
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/plots/"
)

# ---- output names ---------------------------------------------------------
# Every output is named <stem>_<OUTPUT_LABEL>.<ext>, so it can never collide
# with the absolute-differences notebook's files of the same stem.
OUTPUT_LABEL = "relative_differences"


def _output_path(stem, ext):
    return os.path.join(PLOT_OUTPUT, f"{stem}_{OUTPUT_LABEL}.{ext}")


NEONATAL_PLOT_PATH = _output_path("neonatal_superregion_scenarios", "png")
NEONATAL_TABLE_PATH = _output_path("neonatal_cumulative_scenario_differences", "csv")
NEONATAL_GLOBAL_PATH = _output_path("neonatal_global_deaths", "png")
NEONATAL_CSV_PATH = _output_path("neonatal_mortality_mean_estimates", "csv")
U5M_PLOT_PATH = _output_path("child_mortality_superregion_scenarios", "png")
U5M_TABLE_PATH = _output_path("child_mortality_cumulative_scenario_differences", "csv")
U5M_GLOBAL_PATH = _output_path("child_mortality_global_deaths", "png")
U5M_CSV_PATH = _output_path("child_mortality_mean_estimates", "csv")

# ---- FHS indicators -------------------------------------------------------
FHS_ROOT = "/mnt/share/forecasting/data/32/future/death/future_death_aggregated_rate_s130v156_q0_indicators/"
U5M_PATH = os.path.join(FHS_ROOT, "u5mr_5q0.csv")
NMR_PATH = os.path.join(FHS_ROOT, "nmr_28dayq0.csv")
# FHS publishes deaths per 1000 livebirths; the pipeline works in probability
FHS_RATE_PER = 1000.0

# ---- RRA results ----------------------------------------------------------
# The ORIGINAL model draws - inference output, before the residual step. The
# residual step adds its residual and GBD anchor identically to every
# scenario, then clips every draw at zero, which flattens the scenario gap
# wherever the residual has dragged values negative. The original draws are
# never clipped, and as logistic predictions they are never zero, so the
# ratio below is always defined.
NMR_RRA_ROOT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_10_08.01/"
# 5q28; the forecast step has already combined its seven strata into age group 1
CM_RRA_ROOT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_10_16.01/"

# ---- shared inputs for aggregation ---------------------------------------
# Live births: the weight for both mortality indicators. 28q0 and 5q0 are
# probabilities per live birth, so births are what they are aggregated by and
# what turns a rate into a count of deaths. Release 32, scenario 130, matching
# the FHS mortality series above. National only and no sex dimension; see
# load_live_births_weights for how it is split below that.
LIVE_BIRTHS_PATH = "/mnt/share/forecasting/data/32/future/live_births/future_livebirth_s130v97/live_births.nc"
# maternal age groups 15-19 .. 45-49; the file also carries 24, their sum,
# which must be left out or every birth is counted twice
LIVE_BIRTHS_MATERNAL_AGES = [8, 9, 10, 11, 12, 13, 14]

# Population, ages 2 + 3. Used ONLY to split national births across
# subnational locations and sexes, never as a weight in its own right.
FHS_POP_PATH = "/mnt/share/forecasting/data/32/future/population/future_population_s130v41/population.nc"
FHS_HIERARCHY_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/fhs_hierarchy.parquet"
NEONATAL_POP_PARTS = [2, 3]

# ---- keys, labels and analysis settings ----------------------------------
SCEN_KEYS = ["age_group_id", "sex_id", "location_id", "year_id"]
DRAW_IDX = ["location_id", "year_id", "sex_id"]
NEONATAL_AGE_GROUP_ID = 42
UNDER5_AGE_GROUP_ID = 1
BOTH_SEXES_ID = 3  # FHS publishes a Both row; RRA models sexes 1 and 2
GLOBAL_LOCATION_ID = 1
SCENARIO_LABELS = {"reference": "ssp245", "better": "ssp126", "worse": "ssp585"}
REFERENCE_SSP, BETTER_SSP, WORSE_SSP = "ssp245", "ssp126", "ssp585"
SCENARIO_ORDER = ["ssp126", "ssp245", "ssp585"]
SCENARIO_COLOURS = ["#2a78d6", "#52514e", "#eb6834"]
# shown on the figure; the SSP names stay in the data
SCENARIO_DISPLAY = {"ssp126": "RCP 2.6", "ssp245": "RCP 4.5", "ssp585": "RCP 8.5"}
SCENARIO_LEGEND_TITLE = "Representative Control Pathways (RCP) Scenarios"
SCENARIO_PAIRS = [("ssp585", "ssp126"), ("ssp585", "ssp245"), ("ssp245", "ssp126")]
RRA_LEVEL_TO_SSP = {"ref": REFERENCE_SSP, "better": BETTER_SSP, "worse": WORSE_SSP}
FIRST_FORECAST_YEAR = 2024
ANALYSIS_YEARS = [2050, 2100]
NEONATAL_YLABEL = "Neonatal deaths per 1000 livebirths"
U5M_YLABEL = "Under-5 deaths per 1000 livebirths"

# ---- Goalkeepers-format CSV -----------------------------------------------
# The layout of rra_climate_health.csv_creation in the rra-climate-health_2
# clone - columns, labels and location set - copied rather than imported:
# that module pulls in the inference code and its raster dependencies, which
# this script does not otherwise need. Keep the two in step.
CSV_INDICATORS = {  # age_group_id: (rate Indicator, count Indicator, Age)
    NEONATAL_AGE_GROUP_ID: (
        "Neonatal Mortality",
        "Neonatal Mortality counts",
        "Neonatal",
    ),
    UNDER5_AGE_GROUP_ID: ("Under-5 Mortality", "Under-5 Mortality counts", "Under 5"),
}
CSV_UNITS = "Rate per 1,000 live births"
CSV_COUNT_UNITS = "Death counts"
CSV_YEARS = range(2025, 2101)
CSV_SCENARIO_LABELS = {
    "reference": "RCP 4.5 (reference)",
    "better": "RCP 2.6 (optimistic)",
    "worse": "RCP 8.5 (pessimistic)",
}
CSV_SEX_LABELS = {1: "Male", 2: "Female", BOTH_SEXES_ID: "Both sexes"}
# regions that share their name and extent with their super region (North
# Africa and Middle East, South Asia), dropped as in Goalkeepers
DUPLICATE_REGION_IDS = [138, 159]
CSV_COLUMNS = [
    "location_id",
    "Sex",
    "Scenario type",
    "Super region",
    "Region",
    "Location",
    "Year",
    "Indicator",
    "Age",
    "Mean estimate",
    "Indicator units",
]


def _draw_mean(path, name):
    """Mean over the draw columns of an RRA results file, keyed for a merge."""
    d = pd.read_parquet(path).mean(axis=1).rename(name).reset_index()
    return d.drop(columns=[c for c in ("scenario",) if c in d.columns])


def load_neonatal_population():
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


def load_live_births_weights():
    """Live births at every RRA most-detailed location, by year and sex.

    The births file is national only and has no sex dimension, so each
    country's births are divided among its most-detailed (location, sex)
    cells in proportion to their ages 2 + 3 population:

        births[l, s] = births[country] * pop23[l, s] / pop23[country]

    National totals are therefore preserved exactly.
    """
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
    md = hier.loc[
        hier["most_detailed"] == 1, ["location_id", "path_to_top_parent"]
    ].copy()
    # the level-3 ancestor in each location's own path is its country
    md["country_id"] = md["path_to_top_parent"].str.split(",").str[3].astype(int)
    md = md[["location_id", "country_id"]]

    births = (
        xr.open_dataset(LIVE_BIRTHS_PATH)["livebirths"]
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


def rra_nmr_scenario_levels():
    """The three neonatal scenario levels at RRA's own locations."""
    paths = {s: os.path.join(NMR_RRA_ROOT, f"{s}.parquet") for s in SCENARIO_ORDER}
    return (
        _draw_mean(paths[REFERENCE_SSP], "ref")
        .merge(_draw_mean(paths[BETTER_SSP], "better"), on=SCEN_KEYS)
        .merge(_draw_mean(paths[WORSE_SSP], "worse"), on=SCEN_KEYS)
    )


def rra_5q0_draws(scenario):
    """5q0 = 1 - (1 - 28q0) * (1 - 5q28), draw by draw, for one scenario.

    Draw by draw and not on the draw means, because the identity is not
    linear. RRA has no directly usable 5q0: the direct run falls below 28q0
    in 216 locations after 2023, so 5q0 is rebuilt from the two parts.
    """

    def _load(path):
        d = pd.read_parquet(path).reset_index()
        return d.drop(columns=["age_group_id", "scenario"]).set_index(DRAW_IDX)

    neo = _load(os.path.join(NMR_RRA_ROOT, f"{scenario}.parquet"))
    cm = _load(os.path.join(CM_RRA_ROOT, f"{scenario}.parquet")).reindex(neo.index)
    if cm.isna().any().any():
        msg = "28q0 and 5q28 do not cover the same location/year/sex rows"
        raise ValueError(msg)
    return 1.0 - (1.0 - neo) * (1.0 - cm)


def rra_5q0_scenario_levels():
    """The three recombined 5q0 scenario levels, at RRA's own locations."""
    draws = {s: rra_5q0_draws(s).mean(axis=1) for s in SCENARIO_ORDER}
    d = pd.DataFrame(
        {
            "ref": draws[REFERENCE_SSP],
            "better": draws[BETTER_SSP],
            "worse": draws[WORSE_SSP],
        }
    ).reset_index()
    d["age_group_id"] = UNDER5_AGE_GROUP_ID
    return d


def aggregate_scenarios(scenarios, weights, age_group_id):
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
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
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

    def _collapse(frame, keys):
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


def apply_ratios_to_fhs(fhs_path, ratios):
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
    keep = list(fhs.columns.drop(["upper", "lower"])) + ["births"]
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


def counts_from_scenarios(scen):
    """Deaths and live births, at the globe and super regions, Both sexes.

    deaths = rate per 1000 / 1000 * live births. The rates come from FHS at
    each aggregate level, and the births are the most-detailed births summed
    to that same level, so nothing is summed up the hierarchy here.
    """
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
    table_locs = set(hier.loc[hier["level"] < 2, "location_id"])
    out = scen[
        scen["location_id"].isin(table_locs) & (scen["sex_id"] == BOTH_SEXES_ID)
    ].copy()
    out["count"] = out["mean"] / FHS_RATE_PER * out["births"]
    out["rate"] = out["mean"]
    out["scenario"] = out["scenario"].map(SCENARIO_LABELS)
    return out[
        ["location_id", "scenario", "year_id", "count", "births", "rate"]
    ].reset_index(drop=True)


def goalkeepers_csv(scenarios, age_group_id):
    """The scenario series in the Goalkeepers results format.

    One row per metric, location, sex, scenario and year (CSV_YEARS), for the
    global, super region, region and national locations. FHS publishes all
    of them, both sexes included, so the rate is the FHS rate per 1000 as it
    stands; the count is that rate times the live births summed to the same
    location, as in counts_from_scenarios.
    """
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
    locations = hier.query(
        "level <= 3 and location_id not in @DUPLICATE_REGION_IDS"
    )[["location_id", "super_region_name", "region_name", "location_ascii_name"]]
    indicator, count_indicator, age = CSV_INDICATORS[age_group_id]

    df = scenarios[scenarios["year_id"].isin(CSV_YEARS)].merge(
        locations, on="location_id", how="inner"
    )
    missing = set(locations["location_id"]) - set(df["location_id"])
    if missing:
        msg = f"No scenario series for location_ids {sorted(missing)}"
        raise ValueError(msg)
    missing_years = set(CSV_YEARS) - set(df["year_id"])
    if missing_years:
        msg = f"No scenario series for years {sorted(missing_years)}"
        raise ValueError(msg)
    df = df.assign(
        **{
            "Sex": df["sex_id"].map(CSV_SEX_LABELS),
            "Scenario type": df["scenario"].map(CSV_SCENARIO_LABELS),
            "Age": age,
        }
    )
    rates = df.assign(
        **{
            "Indicator": indicator,
            "Mean estimate": df["mean"],
            "Indicator units": CSV_UNITS,
        }
    )
    counts = df.assign(
        **{
            "Indicator": count_indicator,
            "Mean estimate": df["mean"] / FHS_RATE_PER * df["births"],
            "Indicator units": CSV_COUNT_UNITS,
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
        CSV_COLUMNS
    ]


def rra_modeled_counts(agg):
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


def cumulative_difference_table(counts):
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
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
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


def plot_superregion_rates(counts, ylabel, ncol=4):
    """Rate by scenario, one panel per globe/super region, shared y axis."""
    hier = pd.read_parquet(FHS_HIERARCHY_PATH)
    locs = (
        hier[hier["location_id"].isin(counts.location_id.unique())]
        .query("level < 2")
        .sort_values("sort_order")
    )
    nrow = int(np.ceil(len(locs) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), sharex=True, sharey=True
    )
    handles = []
    for ax, (_, loc) in zip(axes.flatten(), locs.iterrows()):
        panel = counts[counts["location_id"] == loc["location_id"]]
        for scen, colour in zip(SCENARIO_ORDER, SCENARIO_COLOURS):
            s = panel[panel["scenario"] == scen].sort_values("year_id")
            (line,) = ax.plot(s["year_id"], s["rate"], color=colour, lw=2)
            if len(handles) < len(SCENARIO_ORDER):
                line.set_label(SCENARIO_DISPLAY[scen])
                handles.append(line)
        ax.set_title(loc["location_name"], fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
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


def _global_cumulative(counts):
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


def _annotate_gap(ax, frame, col, year, label):
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


def _series_label(source, scen):
    """Legend label for one line: FHS (solid) or RRA model (dashed)."""
    if source == "FHS":
        ref = " (reference)" if scen == REFERENCE_SSP else ""
        return f"FHS, {SCENARIO_DISPLAY[scen]}{ref}"
    return f"RRA model, {SCENARIO_DISPLAY[scen]}"


def plot_global_deaths(fhs_counts, rra_counts, deaths_label):
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
    colour = dict(zip(SCENARIO_ORDER, SCENARIO_COLOURS))
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
        ax.grid(True, linestyle=":", alpha=0.6)
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


def run_indicator(name, levels, age_group_id, fhs_path, births, paths, ylabel, deaths_label):
    """Build, tabulate, plot and export one indicator with the relative method."""
    plot_path, table_path, global_path, csv_path = paths
    agg = aggregate_scenarios(levels, births, age_group_id)
    scenarios, clipped = apply_ratios_to_fhs(fhs_path, agg)
    counts = counts_from_scenarios(scenarios)
    table = cumulative_difference_table(counts)

    goalkeepers_csv(scenarios, age_group_id).to_csv(csv_path, index=False)
    fig = plot_superregion_rates(counts, ylabel)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    fig = plot_global_deaths(counts, rra_modeled_counts(agg), deaths_label)
    fig.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n== {name}: rows floored at zero {clipped}")
    for path in paths:
        print(f"  wrote {path}")
    print(table.query("location_name == 'Global'").to_string(index=False))


def main():
    births = load_live_births_weights()
    run_indicator(
        "Neonatal mortality",
        rra_nmr_scenario_levels(),
        NEONATAL_AGE_GROUP_ID,
        NMR_PATH,
        births,
        (
            NEONATAL_PLOT_PATH,
            NEONATAL_TABLE_PATH,
            NEONATAL_GLOBAL_PATH,
            NEONATAL_CSV_PATH,
        ),
        NEONATAL_YLABEL,
        "Neonatal deaths",
    )
    run_indicator(
        "Under-5 mortality",
        rra_5q0_scenario_levels(),
        UNDER5_AGE_GROUP_ID,
        U5M_PATH,
        births,
        (U5M_PLOT_PATH, U5M_TABLE_PATH, U5M_GLOBAL_PATH, U5M_CSV_PATH),
        U5M_YLABEL,
        "Under-5 deaths",
    )


if __name__ == "__main__":
    main()
