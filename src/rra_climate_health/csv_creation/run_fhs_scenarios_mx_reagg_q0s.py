"""FHS q0 scenarios rebuilt from re-aggregated mx, with FHS's life table ax.

Applies the RRA model's scenario effect to FHS's reference mortality rates (mx)
at countries, by detailed under-5 age group, turns them into deaths, builds
every aggregate by summing deaths the way FHS's aggregator does, and then
recalculates q0 from the aggregated mx with FHS's mean age at death (ax), as
FHS's ``child_mortality_probability.py`` does.

Notation, used throughout:

    c   a country
    A   an aggregate location (region, super region or Global)
    a   a detailed under-5 age group (2, 3, 388, 389, 238 or 34)
    s   a scenario (ssp126, ssp245 or ssp585)

Every quantity is also by sex, year and draw; those are left out below.

    mx[c, a, s]      = fhs_mx[c, a] * ratio[c, a, s]
    deaths[c, a, s]  = mx[c, a, s] * population[c, a]
    deaths[A, a, s]  = sum of deaths[c, a, s] over the countries c in A
                       (each region times its GBD population scalar)
    mx[A, a, s]      = deaths[A, a, s] / population[A, a]
    nqx[a]           = mx[a] * nx[a] / (1 + (nx[a] - ax[a]) * mx[a])
                       (ax capped at nx; nx = length of age group a in years)
    q0               = 1 - product over a of (1 - nqx[a])

Neonatal mortality (28q0) is built from age groups 2 and 3 (0-6 and 7-27
days); under-5 mortality (5q0) from 2, 3, 388, 389, 238 and 34 (0-6 days to
2-4 years). ssp245 is the reference, ssp126 the optimistic scenario and ssp585
the pessimistic one.

Step by step:

1. RRA scenario levels at RRA's most-detailed locations (draw means): 28q0
   from the neonatal run and 5q28 from the child mortality run.
2. Each level averaged up to its country c, by sex, weighted by live births,
   and divided by the reference scenario's level at c. This gives two
   scenario ratios per country: a 28q0 ratio and a 5q28 ratio.
3. Scenario mx and deaths at each country c, for each detailed age group a,
   draw by draw. FHS's reference mx is multiplied by the ratio for the part
   of under-5 that a falls in (ratio = 1 in the reference):

       a = 2, 3 (0-27 days)                    mx[c, a] x the 28q0 ratio[c]
       a = 388, 389, 238, 34 (28 days-4 yrs)   mx[c, a] x the 5q28 ratio[c]

   then deaths[c, a] = mx[c, a] x population[c, a].
4. Neonatal and under-5 deaths at each country c: the sum of c's deaths over
   the measure's detailed age groups, draw by draw:

       neonatal deaths[c] = deaths[c, 2] + deaths[c, 3]
       under-5 deaths[c]  = deaths[c, 2] + deaths[c, 3] + deaths[c, 388]
                            + deaths[c, 389] + deaths[c, 238] + deaths[c, 34]

   No ax is needed, and under-5 deaths always include the neonatal deaths.
5. Deaths and population summed from countries c to each aggregate A, and
   across sexes:

       region A       = scalar[A] x sum of deaths[c] over the countries c in A
       super region A = sum of its regions
       Global         = sum of the super regions

   This gives each aggregate's neonatal and under-5 deaths. Kept by detailed
   age group, it also gives mx[A, a] = deaths[A, a] / population[A, a], used
   in step 6.
6. q0 at every location (each country c and aggregate A), draw by draw, from
   its mx and FHS's ax, as FHS's ``calculate_nq0`` does. For each detailed age
   group a:

       mx[a]  mortality rate in a (deaths per person-year), from step 3 or 5
       nx[a]  length of a in years (2: 7/365, 3: 21/365, 388: 0.42,
              389: 0.5, 238: 1, 34: 3)
       ax[a]  for children who die in a, the average time between the start
              of a and their death, in years. For example, Global ax[34] in
              2050 is about 1.5: children who die between ages 2 and 5 die,
              on average, 1.5 years after their 2nd birthday. From FHS's
              life table for its reference forecast
              (FHS scenario 130, version 156: life_expectancy/
              future_life_expectancy_aggregated_s130v156/ax_agg.nc), the
              forecast that is RCP 4.5 here. FHS has no ax for RCP 2.6 or
              RCP 8.5, so this same ax is used in all three scenarios; only
              mx changes between them. Capped at nx[a], as FHS does, so
              that no child is placed past the end of the age group.

   First the probability of dying within a, for those who enter it:

       nqx[a] = mx[a] * nx[a] / (1 + (nx[a] - ax[a]) * mx[a])

   then the probability of dying before the end of the measure's last age
   group, by surviving each age group in turn:

       28q0 = 1 - (1 - nqx[2]) * (1 - nqx[3])
       5q0  = 1 - (1 - nqx[2]) * (1 - nqx[3]) * (1 - nqx[388])
                * (1 - nqx[389]) * (1 - nqx[238]) * (1 - nqx[34])

   Both come from the same scenario mx, so the neonatal part of 5q0 always
   matches 28q0. The CSVs report q0 per 1,000 live births.
7. Deaths and q0 averaged over draws.

Order in the code: steps 4 and 5 run in reverse. The code sums countries to
aggregates first, keeping deaths by detailed age group because q0 needs them
(step 6), and sums the age groups last. Deaths add, so the neonatal and
under-5 deaths are the same either way.

Draw by draw because q0 is not linear in mx or ax: the same calculation on
draw means differs from FHS's published q0 by up to 15%. Done draw by draw,
the reference scenario reproduces FHS's mx at every aggregate and FHS's q0 at
every location, and the run stops if it does not (check_reference_q0).

Writes to ``<output root>/fhs_scenarios_mx_reagg_q0s``, replacing the previous
run's files, for each measure:

- ``<file_stem>_<YYYYMMDD>.csv`` (``neonatal_mortality_<date>.csv``,
  ``under-5_mortality_<date>.csv``, dated the day of the run): q0 per 1,000
  live births and deaths in the Goalkeepers format of ``strun csv_creation``,
  2025-2100, at the global, super region, region and national locations,
  scenarios in the order reference, optimistic, pessimistic. A copy of each is
  also written to ``<output root>/final_csvs``;
- ``<measure>_cumulative_scenario_differences.csv``: cumulative deaths by
  scenario and their differences, at the globe and super regions;
- ``<measure>_superregion_scenarios.png`` and ``<measure>_global_deaths.png``;

plus ``inputs.txt``, listing the results versions and files used.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData

# ---- run settings -----------------------------------------------------------
# Set these to step through the script interactively, or call
# fhs_scenarios_mx_reagg_q0s_main() with no arguments to run it all with them.
# They are also the defaults of the click command, whose options override them.
OUTPUT_ROOT = Path(DEFAULT_ROOT)
NEONATAL_VERSION = "2026_10_08.01"  # neonatal_mortality results version
CHILD_MORTALITY_VERSION = "2026_10_16.01"  # child_mortality (5q28) results version

OUTPUT_DIR = "fhs_scenarios_mx_reagg_q0s"
INPUTS_FILE = "inputs.txt"

# ---- FHS inputs (release 32, scenario 130, version 156) ---------------------
FHS_ROOT = Path("/mnt/share/forecasting/data/32/future")
INPUT_ROOT = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition"
)
# FHS reference mx, draws, every location, sexes 1-3, every age group
FHS_MX_PATH = FHS_ROOT / "death" / "future_death_aggregated_rate_s130v156" / "_all.nc"
# FHS mean age at death (ax), draws, every location (aggregates included),
# sexes 1-3; the copy saved by notebooks/pull_fhs_mx.py from
# life_expectancy/future_life_expectancy_aggregated_s130v156/ax_agg.nc
FHS_AX_PATH = INPUT_ROOT / "fhs_mx" / "ax_agg.nc"
# FHS reference q0 per 1,000 live births: the reference scenario must equal it
FHS_Q0_ROOT = FHS_ROOT / "death" / "future_death_aggregated_rate_s130v156_q0_indicators"
# FHS population, draws, most-detailed locations (and some nationals), sexes 1-2,
# by detailed age group (notebooks/prep_population.py)
POPULATION_PATH = INPUT_ROOT / "population_files" / "population.nc"
# GBD regional population scalars, one per region (notebooks/prep_population.py)
REGION_SCALARS_PATH = INPUT_ROOT / "population_files" / "region_scalars.csv"
# Release 32 population, ages 2 + 3. Used ONLY to split national live births
# across subnational locations and sexes (step 2), never for deaths.
BIRTHS_SPLIT_POP_PATH = (
    FHS_ROOT / "population" / "future_population_s130v41" / "population.nc"
)
BIRTHS_SPLIT_AGE_GROUPS = [2, 3]

PER_1000 = 1000.0
# How far the reference q0 may be from FHS's, relative (floating point only)
REFERENCE_TOLERANCE = 1e-9

# ---- age groups ---------------------------------------------------------------
NEONATAL_AGES = [2, 3]  # 0-6 days, 7-27 days
# 1-5 months, 6-11 months, 12-23 months, 2-4 years
POSTNEONATAL_AGES = [388, 389, 238, 34]
UNDER5_AGES = NEONATAL_AGES + POSTNEONATAL_AGES
# Length of each age group in years (nx), as FHS's nx_from_age_group_ids
# returns them from the GBD age spans. FHS's library is not in this env.
NX = xr.DataArray(
    [0.01917808, 0.05753425, 0.42328767, 0.5, 1.0, 3.0],
    dims="age_group_id",
    coords={"age_group_id": UNDER5_AGES},
    name="nx",
)

# Per measure: its RRA/FHS age group, the detailed ages q0 is built from, its
# FHS q0 file, and labels.
MEASURES = {
    "neonatal_mortality": {
        "age_group_id": 42,
        "ages": NEONATAL_AGES,
        "fhs_q0_file": "nmr_28dayq0.csv",
        "file_stem": "neonatal_mortality",
        "ylabel": "Neonatal deaths per 1,000 live births",
        "deaths_label": "Neonatal deaths",
    },
    "child_mortality": {
        "age_group_id": 1,
        "ages": UNDER5_AGES,
        "fhs_q0_file": "u5mr_5q0.csv",
        "file_stem": "under-5_mortality",
        "ylabel": "Under-5 deaths per 1,000 live births",
        "deaths_label": "Under-5 deaths",
    },
}

# Units of the q0 rows in the CSV (run_csv_creation's own label is "Rate per
# 1,000 live births"; q0 is a probability, so it is labelled as deaths here)
Q0_UNITS = "Deaths per 1,000 live births"
# A copy of each CSV also goes here, under <output root>
FINAL_CSV_DIR = "final_csvs"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# ---- keys, labels and analysis settings ----------------------------------
KEYS = ["location_id", "sex_id", "year_id"]
DRAW_IDX = ["location_id", "year_id", "sex_id"]
GLOBAL_LOCATION_ID = 1
# Hierarchy levels: the tables and plots cover the globe and super regions
# (levels 0-1); the CSV adds regions and countries (levels 2-3).
SUPER_REGION_LEVEL = 1
COUNTRY_LEVEL = NATIONAL_LEVEL = 3
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
# Hierarchy helpers
# =============================================================================


def most_detailed_countries(hier: pd.DataFrame) -> pd.Series:
    """The country (level-3 ancestor) of each most-detailed location.

    Countries without subnationals are their own most-detailed location.
    """
    md = hier.loc[hier["most_detailed"] == 1, ["location_id", "path_to_top_parent"]]
    country = md["path_to_top_parent"].str.split(",").str[COUNTRY_LEVEL].astype(int)
    return pd.Series(country.to_numpy(), index=md["location_id"], name="country_id")


def country_ids(hier: pd.DataFrame) -> list[int]:
    return sorted(hier.loc[hier["level"] == COUNTRY_LEVEL, "location_id"])


# =============================================================================
# Step 1: RRA scenario levels, at RRA's most-detailed locations
# =============================================================================


def _draw_mean(path: Path, name: str) -> pd.DataFrame:
    """Mean over the draw columns of an RRA results file, keyed for a merge."""
    d = pd.read_parquet(path).mean(axis=1).rename(name).reset_index()
    return d.drop(columns=[c for c in ("scenario", "age_group_id") if c in d.columns])


def rra_scenario_levels(results: Path) -> pd.DataFrame:
    """The three scenario levels of one RRA run (28q0 or 5q28), draw means."""
    paths = {s: results / f"{s}.parquet" for s in SCENARIO_ORDER}
    return (
        _draw_mean(paths[REFERENCE_SSP], "ref")
        .merge(_draw_mean(paths[BETTER_SSP], "better"), on=DRAW_IDX)
        .merge(_draw_mean(paths[WORSE_SSP], "worse"), on=DRAW_IDX)
    )


def rra_5q0_draws(nmr_results: Path, cm_results: Path, scenario: str) -> pd.DataFrame:
    """5q0 = 1 - (1 - 28q0) * (1 - 5q28), draw by draw, for one scenario.

    Used only for RRA's own deaths on the Global plot.
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
    return pd.DataFrame(
        {
            "ref": draws[REFERENCE_SSP],
            "better": draws[BETTER_SSP],
            "worse": draws[WORSE_SSP],
        }
    ).reset_index()


# =============================================================================
# Step 2: RRA levels to countries with live births, and the scenario ratios
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
    """
    md = most_detailed_countries(hier).reset_index()

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


def country_scenario_ratios(
    levels: pd.DataFrame, births: pd.DataFrame, hier: pd.DataFrame
) -> pd.DataFrame:
    """RRA's scenario ratios at every country, by sex.

        level[c, s] = sum over l in c of level[l, s] * births[l] / sum of births[l]
        ratio[c, s] = level[c, s] / level[c, reference]

    Live births weight the 5q28 levels too; the survivors to 28 days they
    apply to differ from births by the neonatal deaths only.
    """
    md = most_detailed_countries(hier).reset_index()
    d = levels.merge(births, on=DRAW_IDX, how="inner").merge(
        md, on="location_id", how="inner"
    )
    cols = ["ref", "better", "worse"]
    weighted = d[cols].mul(d["births"], axis=0)
    weighted[["country_id", "sex_id", "year_id", "births"]] = d[
        ["country_id", "sex_id", "year_id", "births"]
    ]
    out = weighted.groupby(["country_id", "sex_id", "year_id"], as_index=False)[
        [*cols, "births"]
    ].sum()
    out[cols] = out[cols].div(out["births"], axis=0)
    if (out["ref"] <= 0).any():
        msg = "RRA reference is zero or negative somewhere; the ratio is undefined"
        raise ValueError(msg)
    out["ratio_better"] = out["better"] / out["ref"]
    out["ratio_worse"] = out["worse"] / out["ref"]
    return out.rename(columns={"country_id": "location_id"})[
        [*KEYS, "ratio_better", "ratio_worse"]
    ]


def scenario_ratio_array(
    ratios: pd.DataFrame, countries: list[int], years: list[int]
) -> xr.DataArray:
    """One RRA ratio table as (scenario, location_id, sex_id, year_id); 1 in the reference."""
    r = ratios.set_index(KEYS)[["ratio_better", "ratio_worse"]].to_xarray()
    r = r.sel(location_id=countries, year_id=years, sex_id=[1, 2])
    if not np.isfinite(r.to_array()).all():
        msg = "RRA ratios are missing for some country, sex or year"
        raise ValueError(msg)
    return xr.concat(
        [xr.ones_like(r["ratio_better"]), r["ratio_better"], r["ratio_worse"]],
        dim=pd.Index(list(SCENARIO_SSP), name="scenario"),
    )


def age_ratio_array(neonatal: xr.DataArray, postneonatal: xr.DataArray) -> xr.DataArray:
    """The ratio by detailed age: 28q0's for ages 2-3, 5q28's for ages 388-34."""
    return xr.concat(
        [neonatal] * len(NEONATAL_AGES) + [postneonatal] * len(POSTNEONATAL_AGES),
        dim=pd.Index(UNDER5_AGES, name="age_group_id"),
    )


# =============================================================================
# Step 3: scenario mx and deaths at countries, draw by draw
# =============================================================================


def load_fhs_mx(countries: list[int], years: list[int]) -> xr.DataArray:
    """FHS reference mx draws at countries, males and females, detailed ages."""
    da = xr.open_dataarray(FHS_MX_PATH)
    if "scenario" in da.dims:
        da = da.squeeze("scenario", drop=True)
    return da.sel(
        location_id=countries, sex_id=[1, 2], year_id=years, age_group_id=UNDER5_AGES
    ).load()


def load_country_population(
    hier: pd.DataFrame, countries: list[int], years: list[int]
) -> xr.DataArray:
    """Population draws by detailed age at every country, males and females.

    Summed from the most-detailed locations (the file also has some nationals,
    which are not used, so every country is built the same way).
    """
    pop = (
        xr.open_dataset(POPULATION_PATH)["draws"]
        .sel(age_group_id=UNDER5_AGES, year_id=years, sex_id=[1, 2])
        .load()
    )
    country = most_detailed_countries(hier)
    pop = pop.sel(location_id=country.index.to_numpy())
    groups = xr.DataArray(
        country.to_numpy(), dims="location_id", coords={"location_id": pop.location_id}
    ).rename("country_id")
    pop = pop.groupby(groups).sum().rename(country_id="location_id")
    return pop.sel(location_id=countries)


def country_deaths(
    fhs_mx: xr.DataArray, ratio: xr.DataArray, population: xr.DataArray
) -> xr.Dataset:
    """Deaths and population at countries, draw by draw, in each scenario.

    mx[c, a, s]     = fhs_mx[c, a] * ratio[c, a, s]
    deaths[c, a, s] = mx[c, a, s] * population[c, a]
    """
    mx = fhs_mx * ratio
    return xr.Dataset({"deaths": mx * population, "population": population})


# =============================================================================
# Step 5: aggregate deaths and population as FHS's aggregator does
# =============================================================================


def load_region_scalars() -> pd.Series:
    """Regional population scalar by region location_id (prep_population.py)."""
    return pd.read_csv(REGION_SCALARS_PATH).set_index("location_id")["mean"]


def _sum_to_parent(ds: xr.Dataset, parent: pd.Series) -> xr.Dataset:
    """Sum every location into its parent (parent indexed by location_id)."""
    ids = parent.reindex(ds["location_id"].to_numpy())
    if ids.isna().any():
        msg = f"No parent for location_ids {sorted(ids[ids.isna()].index)}"
        raise ValueError(msg)
    groups = xr.DataArray(
        ids.astype(int).to_numpy(),
        dims="location_id",
        coords={"location_id": ds["location_id"]},
    ).rename("parent_id")
    return ds.groupby(groups).sum().rename(parent_id="location_id")


def aggregate_counts(
    country: xr.Dataset, hier: pd.DataFrame, scalars: pd.Series
) -> xr.Dataset:
    """Deaths and population at every level and both sexes, draw by draw.

        region       = scalar[region] * sum of its countries
        super region = sum of its regions
        Global       = sum of the super regions
        both sexes   = male + female, at every location

    The scalar multiplies deaths and population alike, so it cancels in mx.
    """
    loc = hier.set_index("location_id")
    regions = _sum_to_parent(country, loc["region_id"])
    scalar = xr.DataArray(
        scalars.reindex(regions["location_id"].to_numpy()).fillna(1.0).to_numpy(),
        dims="location_id",
        coords={"location_id": regions["location_id"]},
    )
    regions = regions * scalar
    super_regions = _sum_to_parent(regions, loc["super_region_id"])
    world = super_regions.sum("location_id").expand_dims(
        location_id=[GLOBAL_LOCATION_ID]
    )
    by_sex = xr.concat([country, regions, super_regions, world], dim="location_id")
    both = by_sex.sum("sex_id").expand_dims(sex_id=[BOTH_SEXES_ID])
    return xr.concat([by_sex, both], dim="sex_id")


# =============================================================================
# Steps 4, 6 and 7: deaths by measure, q0 from the aggregated mx and FHS's ax
# =============================================================================


def load_fhs_ax(locations: np.ndarray, years: list[int]) -> xr.DataArray:
    """FHS mean age at death draws, detailed under-5 ages, capped at nx.

    The cap is FHS's own (child_mortality_probability.py), for implausibly
    large ax values.
    """
    ax = xr.open_dataarray(FHS_AX_PATH)
    if "scenario" in ax.dims:
        ax = ax.squeeze("scenario", drop=True)
    ax = ax.sel(
        location_id=locations,
        sex_id=[1, 2, BOTH_SEXES_ID],
        year_id=years,
        age_group_id=UNDER5_AGES,
    ).load()
    return xr.where(ax > NX, NX, ax)


def calculate_nq0(mx: xr.DataArray, ax: xr.DataArray, ages: list[int]) -> xr.DataArray:
    """nq0 = 1 - prod over the ages of (1 - nqx), as FHS's calculate_nq0.

    nqx = mx * nx / (1 + (nx - ax) * mx)
    """
    m = mx.sel(age_group_id=ages)
    a = ax.sel(age_group_id=ages)
    n = NX.sel(age_group_id=ages)
    nqx = (m * n) / (1 + (n - a) * m)
    return 1 - (1 - nqx).prod("age_group_id")


def measure_results(
    agg: xr.Dataset, ax: xr.DataArray, measure: str, hier: pd.DataFrame
) -> pd.DataFrame:
    """q0 per 1,000 live births and deaths for one measure: means over draws.

    One row per location, sex, year and scenario, at every level.
    """
    ages = MEASURES[measure]["ages"]
    mx = agg["deaths"] / agg["population"]
    q0 = calculate_nq0(mx, ax, ages).mean("draw") * PER_1000
    counts = agg.sel(age_group_id=ages).sum("age_group_id").mean("draw")
    out = (
        xr.Dataset(
            {"q0": q0, "deaths": counts["deaths"], "population": counts["population"]}
        )
        .to_dataframe()
        .reset_index()[[*KEYS, "scenario", "q0", "deaths", "population"]]
    )
    out["level"] = out["location_id"].map(hier.set_index("location_id")["level"])
    return out


def check_reference_q0(res: pd.DataFrame, measure: str) -> float:
    """The reference q0 must equal FHS's published q0 at every row.

    Returns the largest relative difference; raises if it is above
    REFERENCE_TOLERANCE or if a row has no FHS q0.
    """
    fhs = pd.read_csv(FHS_Q0_ROOT / MEASURES[measure]["fhs_q0_file"])[
        [*KEYS, "mean"]
    ].rename(columns={"mean": "fhs_q0"})
    ref = res[res["scenario"] == "reference"].merge(fhs, on=KEYS, how="left")
    if ref["fhs_q0"].isna().any():
        missing = sorted(ref.loc[ref["fhs_q0"].isna(), "location_id"].unique())
        msg = f"{measure}: no FHS q0 for location_ids {missing}"
        raise ValueError(msg)
    worst = float((ref["q0"] / ref["fhs_q0"] - 1).abs().max())
    if worst > REFERENCE_TOLERANCE:
        msg = (
            f"{measure}: reference q0 differs from FHS's q0 by up to "
            f"{100 * worst:.4g}% (tolerance {REFERENCE_TOLERANCE:g})"
        )
        raise ValueError(msg)
    return worst


# =============================================================================
# Outputs: CSV, cumulative table and plots
# =============================================================================


def goalkeepers_csv(
    res: pd.DataFrame, measure: str, hier: pd.DataFrame
) -> pd.DataFrame:
    """The scenario series in the Goalkeepers results format.

    One row per metric, location, sex, scenario and year (OUTPUT_YEARS), for
    the global, super region, region and national locations, labelled as in
    ``strun csv_creation``: q0 per 1,000 live births, and deaths.
    """
    locations = hier.loc[
        (hier["level"] <= NATIONAL_LEVEL)
        & ~hier["location_id"].isin(DUPLICATE_REGION_IDS),
        ["location_id", "super_region_name", "region_name", "location_ascii_name"],
    ]
    indicator = INDICATORS[measure][0]
    count_indicator, count_units = COUNT_INDICATORS[measure]

    df = res[res["year_id"].isin(OUTPUT_YEARS)].merge(
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
            "Mean estimate": df["q0"],
            "Indicator units": Q0_UNITS,
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
        msg = "unmapped sex_id or scenario label, or a missing value"
        raise ValueError(msg)
    df["_count"] = df["Indicator"] == count_indicator  # rates before counts
    # scenarios in the order reference, optimistic, pessimistic
    df["_scenario"] = df["scenario"].map({"reference": 0, "better": 1, "worse": 2})
    return df.sort_values(["_count", "location_id", "Sex", "_scenario", "Year"])[
        OUTPUT_COLUMNS
    ]


def table_series(res: pd.DataFrame) -> pd.DataFrame:
    """Deaths and q0 at the globe and super regions, both sexes, for the table and plots."""
    out = res[
        (res["level"] <= SUPER_REGION_LEVEL) & (res["sex_id"] == BOTH_SEXES_ID)
    ].copy()
    out["count"] = out["deaths"]
    out["scenario"] = out["scenario"].map(SCENARIO_SSP)
    return out[["location_id", "scenario", "year_id", "count", "q0"]].reset_index(
        drop=True
    )


def rra_global_deaths(levels: pd.DataFrame, births: pd.DataFrame) -> pd.DataFrame:
    """RRA's own Global deaths, both sexes: sum of q0 x live births over its locations."""
    d = levels.merge(births, on=DRAW_IDX, how="inner")
    rows = []
    for level, ssp in RRA_LEVEL_TO_SSP.items():
        g = (d[level] * d["births"]).groupby(d["year_id"]).sum()
        rows.append(
            pd.DataFrame(
                {
                    "location_id": GLOBAL_LOCATION_ID,
                    "scenario": ssp,
                    "year_id": g.index,
                    "count": g.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


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

    B is always the lower-emission scenario of the pair.
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
    """q0 per 1,000 live births by scenario, one panel per globe/super region."""
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
            (line,) = ax.plot(s["year_id"], s["q0"], color=colour, lw=2)
            if len(handles) < len(SCENARIO_ORDER):
                line.set_label(SCENARIO_DISPLAY[scen])
                handles.append(line)
        ax.set_title(loc["location_name"], fontsize=12)
        ax.grid(visible=True, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=10)
    for ax in axes.flatten()[len(locs) :]:
        ax.set_visible(False)
    # label at the figure's left edge, with the panels moved right to leave a
    # gap between it and the tick labels (tight_layout ignores supylabel)
    fig.supylabel(ylabel, fontsize=12, x=0.005)
    fig.tight_layout(rect=[0.025, 0.08, 1, 1])
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
    """Bracket RCP 8.5 - RCP 2.6 at `year`, labelled in deaths and percent."""
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

    Solid: FHS reference and the FHS scenarios built here. Dashed: RRA's own
    deaths. Brackets give RCP 8.5 - RCP 2.6 at ANALYSIS_YEARS.
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


def write_measure(
    measure: str,
    res: pd.DataFrame,
    rra_levels: pd.DataFrame,
    births: pd.DataFrame,
    hier: pd.DataFrame,
    out_dir: Path,
    final_dir: Path,
    run_date: str,
) -> None:
    """One measure's CSV (and its copy in final_dir), cumulative table and plots."""
    series = table_series(res)
    table = cumulative_difference_table(series, hier)

    csv_name = f"{MEASURES[measure]['file_stem']}_{run_date}.csv"
    csv_path = out_dir / csv_name
    final_path = final_dir / csv_name
    table_path = out_dir / f"{measure}_cumulative_scenario_differences.csv"
    plot_path = out_dir / f"{measure}_superregion_scenarios.png"
    global_path = out_dir / f"{measure}_global_deaths.png"

    csv = goalkeepers_csv(res, measure, hier)
    csv.to_csv(csv_path, index=False)
    csv.to_csv(final_path, index=False)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    fig = plot_superregion_rates(series, MEASURES[measure]["ylabel"], hier)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig = plot_global_deaths(
        series,
        rra_global_deaths(rra_levels, births),
        MEASURES[measure]["deaths_label"],
    )
    fig.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for path in (csv_path, final_path, table_path, plot_path, global_path):
        print(f"  wrote {path}")
    print(table.query("location_name == 'Global'").to_string(index=False))


def fhs_scenarios_mx_reagg_q0s_main(
    output_root: Path = OUTPUT_ROOT,
    neonatal_version: str = NEONATAL_VERSION,
    child_mortality_version: str = CHILD_MORTALITY_VERSION,
) -> Path:
    """Run steps 1-7 for both measures and write every output to OUTPUT_DIR."""
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
    countries = country_ids(hier)
    fhs_years = xr.open_dataarray(FHS_MX_PATH)["year_id"].to_numpy().tolist()
    years = sorted(y for y in fhs_years if y >= FIRST_FORECAST_YEAR)

    # steps 1-2: RRA levels, then country ratios by detailed age
    births = load_live_births_weights(hier)
    nmr_levels = rra_scenario_levels(nmr_results)
    cm_levels = rra_scenario_levels(cm_results)
    ratio = age_ratio_array(
        scenario_ratio_array(
            country_scenario_ratios(nmr_levels, births, hier), countries, years
        ),
        scenario_ratio_array(
            country_scenario_ratios(cm_levels, births, hier), countries, years
        ),
    )

    # step 3: scenario mx and deaths at countries, draw by draw
    country = country_deaths(
        load_fhs_mx(countries, years),
        ratio,
        load_country_population(hier, countries, years),
    )

    # step 5: aggregate deaths and population
    agg = aggregate_counts(country, hier, load_region_scalars())

    # steps 4, 6 and 7: deaths by measure, q0 from the aggregated mx and FHS's ax,
    # means over draws
    ax = load_fhs_ax(agg["location_id"].to_numpy(), years)
    results = {m: measure_results(agg, ax, m, hier) for m in MEASURES}
    for m, res in results.items():
        worst = check_reference_q0(res, m)
        print(f"{m}: reference q0 vs FHS q0, largest |relative difference| {worst:.1e}")

    out_dir = output_root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / FINAL_CSV_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=LOCAL_TZ)
    written = now.strftime("%Y-%m-%d %H:%M %Z")
    run_date = now.strftime("%Y%m%d")  # dates the CSV file names
    (out_dir / INPUTS_FILE).write_text(
        f"written:                    {written}\n"
        f"neonatal_mortality results: {nmr_results}\n"
        f"child_mortality results:    {cm_results}\n"
        f"FHS mx:                     {FHS_MX_PATH}\n"
        f"FHS ax:                     {FHS_AX_PATH}\n"
        f"FHS q0 (reference check):   {FHS_Q0_ROOT}\n"
        f"population (deaths):        {POPULATION_PATH}\n"
        f"regional scalars:           {REGION_SCALARS_PATH}\n"
        f"live births (RRA weights):  {LIVE_BIRTHS_FILEPATH}\n"
        f"population (births split):  {BIRTHS_SPLIT_POP_PATH}\n"
    )

    rra_levels = {
        "neonatal_mortality": nmr_levels,
        "child_mortality": rra_5q0_scenario_levels(nmr_results, cm_results),
    }
    for m in MEASURES:
        print(f"\n== {m}")
        write_measure(
            m, results[m], rra_levels[m], births, hier, out_dir, final_dir, run_date
        )
    return out_dir


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
    help="child_mortality (5q28) results version whose inference draws to use.",
)
def run_fhs_scenarios_mx_reagg_q0s(
    output_root: str,
    neonatal_version: str,
    child_mortality_version: str,
) -> None:
    """FHS neonatal and under-5 q0 scenarios from re-aggregated mx and FHS's ax."""
    out_dir = fhs_scenarios_mx_reagg_q0s_main(
        Path(output_root), neonatal_version, child_mortality_version
    )
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    run_fhs_scenarios_mx_reagg_q0s()
