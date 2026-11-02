"""Vetting plots for the combined final CSV.

Reads FINAL_CSV_PATH, the output of `strun concat_final_csvs` in the
rra-climate-health_2 clone, and vets every indicator and units in it (e.g.
the under-5 mortality rate and the under-5 death counts), for all of the
indicator's age groups combined. For each, writes a PDF with one panel per
location (Global, super regions, regions, then countries, each in the
Goalkeepers location order; nine per page), lines by sex and scenario, and
prints how many locations drop to 0. All of them are also written to one
interactive HTML page, VETTING_HTML_PATH, which can be filtered by indicator
and by rate or count, and lists the source files and the locations at 0.

Script version of 2026_10_28_csv_explore.ipynb. Run with the gbd_env python:

    /ihme/code/central_comp/miniconda/envs/gbd_env/bin/python 2026_10_28_csv_explore.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import EngFormatter  # noqa: E402

PLOT_OUTPUT = (
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/plots/"
)
LOCATION_ORDER_PATH = (
    "/snfs1/Project/goalkeepers/miscellaneous/location_order_375_gbd2019_20201123.csv"
)

# The combined CSV to vet - here the uncrossed one from `strun
# uncross_scenarios`, in the uncrossed_scenarios subfolder - and the list of
# CSVs `strun concat_final_csvs` combined, in the version folder above it.
FINAL_VERSION_DIR = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/"
    "final_csvs/2026_10_30.02"
)
FINAL_CSV_PATH = FINAL_VERSION_DIR / "uncrossed_scenarios" / "combined_mean_estimates.csv"
FINAL_SOURCES_PATH = FINAL_VERSION_DIR / "sources.txt"

# Units of the count rows; every other unit is a rate or prevalence.
COUNT_UNITS = ["Death counts", "Prevalent counts (cases)"]
# The indicator each CSV Indicator belongs to, so its rate and its count sit
# under one button on the page. An Indicator not listed is its own group.
INDICATOR_GROUPS = {
    "Neonatal Mortality": "Neonatal mortality",
    "Neonatal Mortality counts": "Neonatal mortality",
    "Under-5 Mortality": "Under-5 mortality",
    "Under-5 Mortality counts": "Under-5 mortality",
    "Low Birth Weight Prevalence": "Low birth weight",
    "Low Birth Weight cases": "Low birth weight",
    "Anemia Prevalence": "Anemia",
    "Anemia cases": "Anemia",
}

# The interactive page: the template holds the layout and drawing code, and
# the data for every indicator is embedded in place of __DATA__.
VETTING_HTML_TEMPLATE = Path(__file__).with_name("scenario_vetting_plots_template.html")
VETTING_HTML_PATH = PLOT_OUTPUT + "scenario_vetting_plots.html"

SEX_COLORS = {"Both sexes": "#2a78d6", "Female": "#eb6834", "Male": "#1baf7a"}
SCENARIO_STYLES = {
    "RCP 4.5 (reference)": "-",
    "RCP 2.6 (optimistic)": "--",
    "RCP 8.5 (pessimistic)": ":",
}
# the page's series order: sex-major, matching SEXES and SCENARIOS in the template
SEXES = list(SEX_COLORS)
SCENARIOS = list(SCENARIO_STYLES)
NROWS, NCOLS = 3, 3
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def vetting_pdf_path(indicator):
    """PLOT_OUTPUT/<indicator>_by_sex.pdf, e.g. under_5_mortality_rate_by_sex.pdf."""
    slug = re.sub(r"[^a-z0-9]+", "_", indicator.lower()).strip("_")
    return PLOT_OUTPUT + f"{slug}_by_sex.pdf"


def all_ages_label(ages):
    """The Age label spanning the widest range, e.g. 15-49 years over 15-19.

    An indicator with a single Age (Birth, Neonatal, Under 5) is that age.
    Row order can't be relied on: uncross_scenarios re-sorts by Age.
    """
    ages = list(ages)
    if len(ages) == 1:
        return ages[0]
    spans = {}
    for age in ages:
        m = re.fullmatch(r"(\d+)-(\d+) years", age)
        if m is None:
            msg = f"Can't tell which of {ages} is all ages combined."
            raise ValueError(msg)
        spans[age] = int(m[2]) - int(m[1])
    return max(spans, key=spans.get)


def split_by_indicator(final):
    """One frame per Indicator and units, in file order, all ages combined."""
    for (indicator, units), df in final.groupby(
        ["Indicator", "Indicator units"], sort=False
    ):
        yield indicator, units, df[df["Age"] == all_ages_label(df["Age"].unique())]


def ordered_location_ids(df, location_order):
    """Global, then super regions, then regions, then countries.

    The tier is read off the CSV's own columns: Global has no super region,
    a super region has no region, and a region is its own Region. Within a
    tier, locations keep their place in location_order; any not listed there
    go last.
    """
    meta = df.drop_duplicates("location_id").set_index("location_id")
    position = {loc: k for k, loc in enumerate(location_order["location_id"])}

    def tier(row):
        if pd.isna(row["Super region"]):
            return 0
        if pd.isna(row["Region"]):
            return 1
        if row["Location"] == row["Region"]:
            return 2
        return 3

    return sorted(
        meta.index,
        key=lambda loc: (tier(meta.loc[loc]), position.get(loc, len(position)), loc),
    )


def plot_mean_estimates_by_sex(df, location_order, out_path):
    """One panel per location, aggregates first, lines by sex and scenario."""
    ordered_locs = ordered_location_ids(df, location_order)
    by_loc = dict(tuple(df.groupby("location_id")))
    indicator = df["Indicator"].iloc[0]
    units = df["Indicator units"].iloc[0]
    is_count = units in COUNT_UNITS

    handles = [
        Line2D([], [], color=color, lw=2, label=sex)
        for sex, color in SEX_COLORS.items()
    ] + [
        Line2D([], [], color=MUTED, lw=1.5, ls=ls, label=scenario)
        for scenario, ls in SCENARIO_STYLES.items()
    ]

    per_page = NROWS * NCOLS
    with PdfPages(out_path) as pdf:
        for start in range(0, len(ordered_locs), per_page):
            page_locs = ordered_locs[start : start + per_page]
            fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11, 8.5), squeeze=False)
            for ax, loc in zip(axes.flat, page_locs):
                d = by_loc[loc]
                for (sex, scenario), g in d.groupby(["Sex", "Scenario type"]):
                    ax.plot(
                        g["Year"],
                        g["Mean estimate"],
                        color=SEX_COLORS[sex],
                        ls=SCENARIO_STYLES[scenario],
                        lw=1.5,
                    )
                ax.set_title(d["Location"].iloc[0], fontsize=9, color=INK)
                ax.set_ylim(bottom=0)
                if is_count:
                    # 1.2M rather than an offset "1e6" above the axis
                    ax.yaxis.set_major_formatter(EngFormatter(sep=""))
                ax.grid(color=GRID, lw=0.5)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color(MUTED)
                ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
            for ax in axes.flat[len(page_locs) :]:
                ax.axis("off")
            fig.suptitle(f"{indicator} ({units})", fontsize=11, color=INK)
            fig.legend(
                handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)


def _round_sig(v, digits=4):
    """v to `digits` significant figures, None for a missing value."""
    if pd.isna(v):
        return None
    return float(f"{v:.{digits}g}")


def pack_for_html(df, location_order):
    """One indicator's series in the shape the vetting page reads.

    Per location: its names, whether it drops to 0, and nine series (sex x
    scenario, sex-major) aligned to `years`, with None where a scenario has no
    value.
    """
    years = list(range(int(df["Year"].min()), int(df["Year"].max()) + 1))
    meta = df.drop_duplicates("location_id").set_index("location_id")
    loc_ids = ordered_location_ids(df, location_order)
    zero_ids = set(df.loc[df["Mean estimate"] == 0, "location_id"])

    wide = df.pivot_table(
        index="location_id", columns=["Sex", "Scenario type", "Year"], values="Mean estimate"
    ).reindex(
        index=loc_ids, columns=pd.MultiIndex.from_product([SEXES, SCENARIOS, years])
    )
    arr = wide.to_numpy().reshape(len(loc_ids), len(SEXES) * len(SCENARIOS), len(years))

    locs = []
    for i, loc in enumerate(loc_ids):
        m = meta.loc[loc]
        locs.append(
            {
                "id": int(loc),
                "n": m["Location"],
                "sr": None if pd.isna(m["Super region"]) else m["Super region"],
                "rg": None if pd.isna(m["Region"]) else m["Region"],
                "z": bool(loc in zero_ids),
                "s": [[_round_sig(v) for v in row] for row in arr[i]],
            }
        )
    first = df.iloc[0]
    return {
        "group": INDICATOR_GROUPS.get(first["Indicator"], first["Indicator"]),
        "metric": "Count" if first["Indicator units"] in COUNT_UNITS else "Rate",
        "indicator": first["Indicator"],
        "age": first["Age"],
        "units": first["Indicator units"],
        "years": years,
        "zero": [loc["n"] for loc in locs if loc["z"]],
        "locs": locs,
    }


def write_vetting_html(packed, out_path):
    """Embed the packed indicators and their sources into the page template."""
    sources = FINAL_SOURCES_PATH.read_text().split() if FINAL_SOURCES_PATH.exists() else []
    data = {"path": str(FINAL_CSV_PATH), "sources": sources, "indicators": packed}
    # "</" inside the JSON would end the <script> block holding it
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    html = VETTING_HTML_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    Path(out_path).write_text(html, encoding="utf-8")


def main():
    location_order = pd.read_csv(LOCATION_ORDER_PATH)
    final = pd.read_csv(FINAL_CSV_PATH)
    zero_locations = {}
    packed = []
    for indicator, units, df in split_by_indicator(final):
        pdf_path = vetting_pdf_path(indicator)
        plot_mean_estimates_by_sex(df, location_order, pdf_path)
        print(f"{indicator} ({units}): {pdf_path}")
        # a location drops to 0 if its mean estimate is 0 in at least one
        # year, sex or scenario
        zero_locations[indicator] = df.query("`Mean estimate` == 0")[
            "location_id"
        ].nunique()
        packed.append(pack_for_html(df, location_order))

    write_vetting_html(packed, VETTING_HTML_PATH)
    print(f"Vetting page: {VETTING_HTML_PATH}")
    print()
    print(pd.Series(zero_locations, name="locations at 0").to_string())


if __name__ == "__main__":
    main()
