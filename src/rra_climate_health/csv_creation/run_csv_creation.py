"""CSV export step: the final LBW and anemia CSVs.

Which script makes which final CSV
----------------------------------
- Low birth weight and anemia: this script (``strun csv_creation``).
- Neonatal and under-5 mortality: NOT this script. Their final CSVs come from
  ``run_fhs_scenarios_mx_reagg_q0s.py``, which applies the RRA model's
  scenario ratios to FHS's forecast death rates (mx), re-aggregates the
  locations, and computes 28q0 and 5q0 from the result. The reference
  scenario there equals FHS's own forecast exactly. See that script's
  docstring.

This script still accepts ``neonatal_mortality`` and ``child_mortality``, but
the output is the RRA model on its own, without FHS. It is not the final
result, and for current child_mortality models it is mislabelled: those models
are fit without the first month of life, so they predict 5q28 (death between
28 days and age 5), not 5q0, while the output says "Under-5 Mortality". The
child growth measures (stunting, wasting, underweight) also still run.

What it does
------------
Runs after the residual step. Reads ``shifted_prevalence.parquet`` for a
results version: the RRA model's prevalence draws by scenario, shifted to
match GBD in the past. Draws are clipped to [0, 1] and averaged, then
aggregated up the FHS location hierarchy for each sex and both sexes
combined. It writes ``mean_estimates.csv`` in the same results folder, in the
Goalkeepers results format (see ``xarray_results_to_plot_csv.py`` in the
goal_keepers repo). Output covers:

- metrics: prevalence and the count;
- scenarios: RCP 4.5 (reference), RCP 2.6 (optimistic), RCP 8.5 (pessimistic),
  in that order;
- years: 2025-2100;
- locations: Global, super region, region and national;
- sexes: male, female and both sexes;
- ages: each modelled age group, plus all of them combined when the measure
  has more than one.

Low birth weight
----------------
Prevalence is the share of live births under 2,500 g. It is weighted by live
births, and the count is LBW births (prevalence x live births). Live births
are FHS's (s130v97). FHS gives them for each country, with no sex split, so
they are shared out among subnationals and sexes in proportion to the
neonatal population (POPULATION_PATH, FHS s130v156, age group 42).

Anemia
------
Prevalence is the share of women aged 15-49 with anemia, modelled in 5-year
age groups (15-19 to 45-49). It is weighted by population (POPULATION_PATH,
FHS s130v156, the same file the mortality counts use). The count is prevalent
cases (prevalence x women aged 15-49); summed over years it is
person-years lived with anemia. Only females aged 15-49 combined are written
(REPORTED_SUBSETS).

Aggregation (both)
------------------
Counts are aggregated as FHS's aggregator does for mortality: each region is
its countries' sum times the region's GBD population scalar (one value per
region, for all ages and both sexes), and super regions and Global are sums
of regions. The scalar multiplies the count and its weight alike, so a
region's prevalence is the same as without it.
"""

from pathlib import Path

import click
import pandas as pd
import xarray as xr

from rra_climate_health import cli_options as clio
from rra_climate_health.constants import AGE_GROUP_AGGREGATES, REFERENCE_SCENARIO
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData

OUTPUT_FILE = "mean_estimates.csv"
OUTPUT_YEARS = range(2025, 2101)

# FHS live births: national only, by maternal age, no sex dimension.
LIVE_BIRTHS_FILEPATH = "/mnt/share/forecasting/data/32/future/live_births/future_livebirth_s130v97/live_births.nc"
LIVE_BIRTHS_MATERNAL_AGES = [8, 9, 10, 11, 12, 13, 14]
# FHS population, the same file the mortality counts use: s130v156, draws,
# most-detailed locations, sexes 1-2, by age group, with age groups 42 and 1
# added by notebooks/prep_population.py. Weights the population-based measures
# and splits national births among subnationals and sexes.
POPULATION_PATH = Path(DEFAULT_ROOT) / "population_files" / "population.nc"
# GBD regional population scalars, one per region (all ages, both sexes), from
# FHS's aggregator via notebooks/prep_population.py.
REGION_SCALARS_PATH = Path(DEFAULT_ROOT) / "population_files" / "region_scalars.csv"
REGION_LEVEL = 2
# Age group whose population splits national births among subnationals and sexes.
NEONATAL_AGE_GROUP_ID = 42
PER_BIRTH_MEASURES = ["neonatal_mortality", "child_mortality", "lbw"]

PREVALENCE_UNITS = "Rate (0-100%)"
MORTALITY_RATE_UNITS = "Rate per 1,000 live births"
CASE_COUNT_UNITS = "Prevalent counts (cases)"
DEATH_COUNT_UNITS = "Death counts"

# Indicator name, units label, and the multiplier that puts a proportion in
# those units. Names follow the Goalkeepers conventions where the indicator
# is a Goalkeepers one.
INDICATORS = {
    "stunting": ("Child Stunting", PREVALENCE_UNITS, 100),
    "wasting": ("Child Wasting", PREVALENCE_UNITS, 100),
    "underweight": ("Child Underweight", PREVALENCE_UNITS, 100),
    "anemia": ("Anemia Prevalence", PREVALENCE_UNITS, 100),
    "lbw": ("Low Birth Weight Prevalence", PREVALENCE_UNITS, 100),
    "neonatal_mortality": ("Neonatal Mortality", MORTALITY_RATE_UNITS, 1000),
    "child_mortality": ("Under-5 Mortality", MORTALITY_RATE_UNITS, 1000),
}

# Indicator name and units label for each measure's counts, following the
# Goalkeepers "Under-5 Mortality counts" convention.
COUNT_INDICATORS = {
    "stunting": ("Child Stunting cases", CASE_COUNT_UNITS),
    "wasting": ("Child Wasting cases", CASE_COUNT_UNITS),
    "underweight": ("Child Underweight cases", CASE_COUNT_UNITS),
    "anemia": ("Anemia cases", CASE_COUNT_UNITS),
    "lbw": ("Low Birth Weight cases", CASE_COUNT_UNITS),
    "neonatal_mortality": ("Neonatal Mortality counts", DEATH_COUNT_UNITS),
    "child_mortality": ("Under-5 Mortality counts", DEATH_COUNT_UNITS),
}

# The only sexes and ages written for a measure, where not all of them are:
# anemia is reported for women of reproductive age only.
REPORTED_SUBSETS = {
    "anemia": {"Sex": ["Female"], "Age": ["15-49 years"]},
}

# Stands in for age_group_id on the rows that combine all of a measure's age
# groups; not a GBD age group.
ALL_AGES_ID = 0

# The label for all of each measure's age groups combined. A measure modelled
# in a single age group is reported under this label alone.
AGE_LABELS = {
    "stunting": "Under 5",
    "wasting": "Under 5",
    "underweight": "Under 5",
    "anemia": "15-49 years",
    "lbw": "Birth",
    "neonatal_mortality": "Neonatal",
    "child_mortality": "Under 5",
}

# Labels for the individual age groups of measures modelled in more than one.
AGE_GROUP_LABELS = {
    2: "Early Neonatal",
    3: "Late Neonatal",
    4: "Post Neonatal",
    5: "1-4 years",
    388: "1-5 months",
    389: "6-11 months",
    238: "12-23 months",
    34: "2-4 years",
    8: "15-19 years",
    9: "20-24 years",
    10: "25-29 years",
    11: "30-34 years",
    12: "35-39 years",
    13: "40-44 years",
    14: "45-49 years",
}

SCENARIO_LABELS = {
    REFERENCE_SCENARIO: "RCP 4.5 (reference)",
    "ssp126": "RCP 2.6 (optimistic)",
    "ssp585": "RCP 8.5 (pessimistic)",
}

BOTH_SEXES_ID = 3
SEX_LABELS = {1: "Male", 2: "Female", BOTH_SEXES_ID: "Both sexes"}

# Regions that share their name and extent with their super region
# (North Africa and Middle East, South Asia), dropped as in Goalkeepers.
DUPLICATE_REGION_IDS = [138, 159]

OUTPUT_COLUMNS = [
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


def load_population(
    locations: list[int],
    age_group_ids,
) -> pd.Series:
    """Population (draw mean) by location_id, year_id, sex_id and age_group_id.

    From POPULATION_PATH. An age group that is not in the file is summed from
    its parts (constants.AGE_GROUP_AGGREGATES).
    """
    pop = (
        xr.open_dataset(POPULATION_PATH)["draws"]
        .sel(location_id=list(locations))
        .mean("draw")
    )
    available = set(pop["age_group_id"].to_numpy().tolist())
    frames = []
    for age in age_group_ids:
        parts = [age] if age in available else AGE_GROUP_AGGREGATES.get(age, [])
        if not parts or not set(parts) <= available:
            message = f"No population for age_group_id {age} in {POPULATION_PATH}."
            raise ValueError(message)
        frame = (
            pop.sel(age_group_id=parts)
            .sum("age_group_id")
            .to_series()
            .rename("population")
            .reset_index()
        )
        frame["age_group_id"] = age
        frames.append(frame)
    return pd.concat(frames).set_index(
        ["location_id", "year_id", "sex_id", "age_group_id"]
    )["population"]


def load_region_scalars() -> pd.Series:
    """GBD regional population scalar by region location_id."""
    return pd.read_csv(REGION_SCALARS_PATH).set_index("location_id")["mean"]


def load_live_births(
    hierarchy: pd.DataFrame,
    locations: list[int],
) -> pd.Series:
    """Live births by most-detailed location_id, year_id and sex_id.

    The births file is national only and has no sex dimension, so each
    country's births are divided among its most-detailed (location, sex)
    cells in proportion to their neonatal population (POPULATION_PATH):

        births[l, s] = births[country] * pop42[l, s] / pop42[country]

    National totals are preserved exactly. Same method as
    load_live_births_weights in notebooks/2026_10_20_presentation_plots.ipynb.
    """
    country_ids = (
        hierarchy.query("location_id in @locations")
        .set_index("location_id")["path_to_top_parent"]
        .str.split(",")
        .str[3]
        .astype(int)
        .rename("country_id")
    )
    births = (
        xr.open_dataset(LIVE_BIRTHS_FILEPATH)["livebirths"]
        .sel(age_group_id=LIVE_BIRTHS_MATERNAL_AGES)
        .sum("age_group_id")
        .mean("draw")
        .squeeze(drop=True)
        .to_series()
        .rename("country_births")
        .rename_axis(index={"location_id": "country_id"})
    )
    pop = (
        load_population(locations, [NEONATAL_AGE_GROUP_ID])
        .droplevel("age_group_id")
        .rename("population")
        .reset_index()
        .join(country_ids, on="location_id")
    )
    # keep the years both files cover
    birth_years = births.index.get_level_values("year_id").unique()
    pop = pop[pop["year_id"].isin(birth_years)]
    pop["country_pop"] = pop.groupby(["country_id", "year_id"])["population"].transform(
        "sum"
    )
    pop = pop.join(births, on=["country_id", "year_id"])
    if pop["country_births"].isna().any():
        missing = pop.loc[pop["country_births"].isna(), "country_id"].unique()
        message = f"No live births for country location_ids {sorted(missing)}."
        raise ValueError(message)
    pop["births"] = pop["country_births"] * pop["population"] / pop["country_pop"]
    return pop.set_index(["location_id", "year_id", "sex_id"])["births"]


def aggregate_means(
    means: pd.Series,
    weights: pd.Series,
    hierarchy: pd.DataFrame,
    scalars: pd.Series,
) -> pd.DataFrame:
    """Weight most-detailed means up the hierarchy, over sex and over age.

    Regions and everything above them carry the GBD regional population
    scalar, as in FHS's aggregator: a most-detailed location's count and
    weight are multiplied by its region's scalar wherever they land in a
    region, super region or Global, so

        region       = scalar[region] * sum of its countries
        super region = sum of its regions
        Global       = sum of the super regions

    ``scalars`` is indexed by region location_id; regions without one use 1.

    ``weights`` is population or live births, indexed by a subset of the
    means' index levels. Returns the count (mean x weight) and the weight,
    summed, by location_id, year_id, scenario, sex_id and age_group_id;
    sex_id includes BOTH_SEXES_ID, and age_group_id is ALL_AGES_ID for all
    age groups combined (the only age when there is one age group). The rate
    is count / weight. Taking the draw mean before aggregating gives the same
    answer as aggregating draws, since the weights are fixed.
    """
    df = (
        means.rename("mean")
        .reset_index()
        .merge(
            weights.rename("weight").reset_index(),
            on=list(weights.index.names),
            how="left",
        )
    )
    if df["weight"].isna().any():
        missing = df.loc[df["weight"].isna(), "location_id"].unique()
        message = f"No weights for location_ids {sorted(missing)}."
        raise ValueError(message)
    df["count"] = df["mean"] * df["weight"]

    by = ["location_id", "year_id", "scenario", "age_group_id", "sex_id"]
    df = df.groupby(by)[["count", "weight"]].sum().reset_index()

    ancestors = (
        hierarchy.set_index("location_id")["path_to_top_parent"]
        .str.split(",")
        .apply(lambda ids: [int(i) for i in ids])
    )
    loc = hierarchy.set_index("location_id")
    df["scalar"] = df["location_id"].map(loc["region_id"]).map(scalars).fillna(1.0)
    df["location_id"] = df["location_id"].map(ancestors)
    df = df.explode("location_id")
    df["location_id"] = df["location_id"].astype(int)
    # the scalar applies at regions and above, not at countries or subnationals
    scaled = df["location_id"].map(loc["level"]) <= REGION_LEVEL
    df.loc[scaled, ["count", "weight"]] = df.loc[scaled, ["count", "weight"]].mul(
        df.loc[scaled, "scalar"], axis=0
    )
    df = df.drop(columns="scalar")

    by_sex = df.groupby(by)[["count", "weight"]].sum()
    both_sexes = (
        df.groupby(by[:-1])[["count", "weight"]]
        .sum()
        .assign(sex_id=BOTH_SEXES_ID)
        .set_index("sex_id", append=True)
    )
    agg = pd.concat([by_sex, both_sexes])

    all_ages = (
        agg.groupby([b for b in by if b != "age_group_id"])
        .sum()
        .assign(age_group_id=ALL_AGES_ID)
        .set_index("age_group_id", append=True)
        .reorder_levels(by)
    )
    if agg.index.get_level_values("age_group_id").nunique() == 1:
        return all_ages
    return pd.concat([agg, all_ages])


def format_results(
    aggregated: pd.DataFrame,
    hierarchy: pd.DataFrame,
    measure: str,
) -> pd.DataFrame:
    """Label aggregated rates and counts in the Goalkeepers results format."""
    indicator, units, scale = INDICATORS[measure]
    count_indicator, count_units = COUNT_INDICATORS[measure]
    age_labels = {**AGE_GROUP_LABELS, ALL_AGES_ID: AGE_LABELS[measure]}

    locations = hierarchy.query(
        "level <= 3 and location_id not in @DUPLICATE_REGION_IDS"
    )[["location_id", "super_region_name", "region_name", "location_ascii_name"]]

    df = aggregated.reset_index().merge(locations, on="location_id", how="inner")
    df = df.assign(
        **{
            "Sex": df["sex_id"].map(SEX_LABELS),
            "Scenario type": df["scenario"].map(lambda s: SCENARIO_LABELS.get(s, s)),
            "Age": df["age_group_id"].map(age_labels),
        }
    )
    if df["Age"].isna().any():
        missing = sorted(df.loc[df["Age"].isna(), "age_group_id"].unique())
        message = (
            f"No age label for age_group_ids {missing}; add them to AGE_GROUP_LABELS."
        )
        raise ValueError(message)
    rates = df.assign(
        **{
            "Indicator": indicator,
            "Mean estimate": df["count"] / df["weight"] * scale,
            "Indicator units": units,
        }
    )
    counts = df.assign(
        **{
            "Indicator": count_indicator,
            "Mean estimate": df["count"],
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
    for column, keep in REPORTED_SUBSETS.get(measure, {}).items():
        df = df[df[column].isin(keep)]
    # rates before counts, all ages combined before the age groups, and
    # scenarios in the order reference, optimistic, pessimistic
    df["_count"] = df["Indicator"] == count_indicator
    df["_age"] = df["age_group_id"] != ALL_AGES_ID
    df["_scenario"] = df["Scenario type"].map(
        {label: k for k, label in enumerate(SCENARIO_LABELS.values())}
    )
    return df.sort_values(
        [
            "_count",
            "location_id",
            "_age",
            "age_group_id",
            "Sex",
            "_scenario",
            "Year",
        ]
    )[OUTPUT_COLUMNS]


def get_mean_estimates(
    cm_data: ClimateMalnutritionData,
    measure: str,
    results_version: str,
) -> pd.DataFrame:
    hierarchy = cm_data.load_fhs_hierarchy()
    draws = cm_data.load_shifted_prevalence(results_version)

    most_detailed = set(hierarchy.query("most_detailed == 1")["location_id"])
    draw_locs = set(draws.index.get_level_values("location_id"))
    if missing := sorted(most_detailed - draw_locs):
        print(f"Warning: no draws for most-detailed location_ids {missing}")
    if extra := sorted(draw_locs - most_detailed):
        print(
            f"Warning: dropping location_ids not most-detailed in the hierarchy {extra}"
        )
        draws = draws.query("location_id in @most_detailed")

    years = draws.index.get_level_values("year_id")
    if missing_years := sorted(set(OUTPUT_YEARS) - set(years)):
        message = f"No draws for years {missing_years}."
        raise ValueError(message)
    draws = draws[years.isin(OUTPUT_YEARS)]

    means = draws.clip(0, 1).mean(axis=1)
    locations = sorted(draw_locs & most_detailed)
    if measure in PER_BIRTH_MEASURES:
        weights = load_live_births(hierarchy, locations)
    else:
        weights = load_population(
            locations, means.index.get_level_values("age_group_id").unique()
        )

    aggregated = aggregate_means(means, weights, hierarchy, load_region_scalars())
    return format_results(aggregated, hierarchy, measure)


def csv_creation_main(
    output_root: Path,
    measure: str,
    results_version: str,
) -> Path:
    cm_data = ClimateMalnutritionData(output_root / measure)
    out_path = cm_data.results / results_version / OUTPUT_FILE
    get_mean_estimates(cm_data, measure, results_version).to_csv(out_path, index=False)
    return out_path


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
def run_csv_creation(
    output_root: str,
    measure: str,
    results_version: str,
) -> None:
    """Write a results version's mean estimates to a Goalkeepers-format CSV."""
    cm_data = ClimateMalnutritionData(Path(output_root) / measure)
    # Fail fast with a clear message if the residual step hasn't been run yet.
    draws_path = cm_data.shifted_prevalence_path(results_version)
    if not draws_path.exists():
        message = (
            f"No shifted prevalence found at {draws_path}. Run the residual step "
            f"for {measure} results version {results_version} first."
        )
        raise click.ClickException(message)

    print(f"Writing mean estimates for {measure}, results version {results_version}")
    out_path = csv_creation_main(Path(output_root), measure, results_version)
    print(f"CSV written to {out_path}")
