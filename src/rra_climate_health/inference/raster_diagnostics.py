"""Coalesce per-draw prediction rasters and plot raster diff diagnostics.

When inference runs with ``--save-rasters``, each inference task writes its
prediction raster for the last GBD year and the last forecast year.  The last
forecast year gets one raster per draw, which is a lot of disk for data whose
only use is a mean map, so the ``coalesce_rasters`` task averages the draws of
one (scenario, age, sex) into a single mean raster and then deletes the
per-draw files.  The ``raster_diagnostics`` task then renders two diff maps
for one representative age/sex stratum (age and sex are just categorical terms
in the model, and there are no population forecast rasters to aggregate over):

- ``raster_scenario_diff.png``: last forecast year, worst minus best scenario.
- ``raster_year_diff.png``: reference scenario, last forecast year minus last
  GBD year.  Historical years are only run for draw 0, so the last GBD year's
  raster is the ``_0`` draw file used directly, and it is never deleted.

Both tasks are idempotent and can be run standalone against any results
version that has rasters, including versions coalesced by hand before this
step existed.
"""

from pathlib import Path

import click
import numpy as np
import rasterra as rt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from rra_climate_health import cli_options as clio
from rra_climate_health.constants import (
    LAST_FORECAST_YEAR,
    LAST_GBD_YEAR,
    RASTER_DIAGNOSTIC_AGE_SEX,
    RASTER_DIAGNOSTIC_LABELS,
    REFERENCE_SCENARIO,
)
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData

# The scenario pair contrasted in the scenario diff map.
SCENARIO_DIFF_MINUEND = "ssp585"
SCENARIO_DIFF_SUBTRAHEND = "ssp126"

SCENARIO_DISPLAY_NAMES = {
    "ssp126": "SSP1 2.6",
    "ssp245": "SSP2 4.5",
    "ssp585": "SSP5 8.5",
}


def coalesce_raster_draws_main(
    output_dir: Path,
    measure: str,
    results_version: str,
    cmip6_scenario: str,
    age_group_id: int | str,
    sex_id: int,
) -> None:
    """Average the last-forecast-year draw rasters of one (scenario, age, sex)
    into a mean raster, then delete the per-draw rasters to free space.

    Draw files are only deleted after the mean raster has been written and
    verifies by reading back, so an interrupted run can always be rerun: a
    missing or unreadable mean is recomputed from the draws, and a valid mean
    with leftover draw files just finishes the deletion.
    """
    cm_data = ClimateMalnutritionData(output_dir / measure)
    results_spec = cm_data.load_results_specification(results_version)
    if not results_spec.save_rasters:
        print(
            f"Note: results version {results_version} was not run with "
            "--save-rasters (or predates the flag); proceeding on whatever "
            "raster files are present."
        )
    year = LAST_FORECAST_YEAR
    mean_path = cm_data.mean_raster_results_path(
        results_version, cmip6_scenario, year, age_group_id, sex_id
    )
    draw_paths = [
        cm_data.raster_results_path(
            results_version, cmip6_scenario, year, age_group_id, sex_id, draw
        )
        for draw in range(results_spec.draws)
    ]

    if mean_path.exists():
        try:
            rt.load_raster(mean_path)
            mean_is_valid = True
        except Exception:  # noqa: BLE001
            mean_is_valid = False
        if mean_is_valid:
            leftover = [p for p in draw_paths if p.exists()]
            for path in leftover:
                path.unlink()
            print(
                f"Mean raster {mean_path.name} already exists; "
                f"deleted {len(leftover)} leftover draw rasters."
            )
            return
        print(f"Mean raster {mean_path.name} exists but is unreadable; recomputing.")

    missing = [p for p in draw_paths if not p.exists()]
    if missing:
        msg = (
            f"Cannot coalesce {cmip6_scenario}/{age_group_id}/{sex_id} for "
            f"{measure} {results_version}: {len(missing)} of {len(draw_paths)} "
            f"draw rasters are missing (e.g. {missing[0].name}) and no valid "
            "mean raster exists."
        )
        raise FileNotFoundError(msg)

    print(f"Averaging {len(draw_paths)} draw rasters into {mean_path.name}")
    mean_raster = 0
    for path in draw_paths:
        mean_raster += rt.load_raster(path)
    mean_raster /= len(draw_paths)
    cm_data.save_mean_raster_results(
        mean_raster, results_version, cmip6_scenario, year, age_group_id, sex_id
    )
    # Verify the mean reads back before deleting anything.
    rt.load_raster(mean_path)
    for path in draw_paths:
        path.unlink()
    print(f"Deleted {len(draw_paths)} draw rasters.")


def plot_raster_diff(data: np.ndarray, label: str, output_path: Path) -> None:
    """Render a diverging map of a raster difference, with 0 mapped to white."""
    finite = np.isfinite(data)
    if not finite.any():
        print(f"No finite values to plot for {output_path.name}; skipping.")
        return

    # crop to the bounding box of non-NaN cells (drops the empty polar bands)
    rows, cols = np.where(finite)
    data = data[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    # color limits, with 0 kept as an endpoint
    vmin, vmax = np.nanpercentile(data, [1, 99])
    vmin, vmax = min(vmin, 0.0), max(vmax, 0.0)

    # slice RdBu_r so that 0 is white, without carrying unused colors
    m = max(-vmin, vmax)
    if m == 0:
        print(f"Difference is zero everywhere for {output_path.name}; skipping.")
        return
    cmap = LinearSegmentedColormap.from_list(
        "slice",
        colormaps["RdBu_r"](
            np.linspace((vmin + m) / (2 * m), (vmax + m) / (2 * m), 256)
        ),
    )
    cmap.set_bad("#ececec")

    # figure matches the map's aspect, full slide width
    h, w = data.shape
    CBAR_FRAC = 0.12  # share of figure width given to the colorbar
    FIG_W = 13.333

    map_w = FIG_W * (1 - CBAR_FRAC)  # map's width in inches
    fig = Figure(figsize=(FIG_W, map_w * h / w), dpi=200)

    ax = fig.add_axes([0, 0, 1 - CBAR_FRAC, 1])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    CBAR_H = 0.78  # fraction of figure height
    cax = fig.add_axes([1 - CBAR_FRAC + 0.02, (1 - CBAR_H) / 2, 0.016, CBAR_H])

    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label(label, fontsize=13, labelpad=12)
    cbar.ax.tick_params(labelsize=12, length=3, pad=4)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(output_path, dpi=200, facecolor="white")
    print(f"Wrote {output_path}")


def raster_diagnostics_main(
    output_dir: Path,
    measure: str,
    results_version: str,
) -> None:
    """Plot the scenario diff and year diff maps for one age/sex stratum.

    Each map is guarded on the existence of its input rasters rather than the
    results spec, so a partially populated version still produces whatever is
    producible.
    """
    cm_data = ClimateMalnutritionData(output_dir / measure)
    results_spec = cm_data.load_results_specification(results_version)

    preferred = RASTER_DIAGNOSTIC_AGE_SEX.get(measure)
    if (
        preferred is not None
        and preferred[0] in results_spec.age_groups
        and preferred[1] in results_spec.sex_ids
    ):
        age_group_id, sex_id = preferred
    else:
        age_group_id = results_spec.age_groups[0]
        sex_id = results_spec.sex_ids[0]
    print(f"Plotting raster diffs for age group {age_group_id}, sex {sex_id}")

    label = RASTER_DIAGNOSTIC_LABELS.get(measure, measure.replace("_", " ").title())
    results_dir = cm_data.results / results_version

    minuend_path = cm_data.mean_raster_results_path(
        results_version, SCENARIO_DIFF_MINUEND, LAST_FORECAST_YEAR, age_group_id, sex_id
    )
    subtrahend_path = cm_data.mean_raster_results_path(
        results_version,
        SCENARIO_DIFF_SUBTRAHEND,
        LAST_FORECAST_YEAR,
        age_group_id,
        sex_id,
    )
    if minuend_path.exists() and subtrahend_path.exists():
        diff = rt.load_raster(minuend_path) - rt.load_raster(subtrahend_path)
        plot_raster_diff(
            diff.to_numpy().astype(float) * 1000,
            f"Difference in {label}, year {LAST_FORECAST_YEAR}\n"
            f"{SCENARIO_DISPLAY_NAMES[SCENARIO_DIFF_MINUEND]} - "
            f"{SCENARIO_DISPLAY_NAMES[SCENARIO_DIFF_SUBTRAHEND]}",
            results_dir / "raster_scenario_diff.png",
        )
    else:
        missing = [p.name for p in (minuend_path, subtrahend_path) if not p.exists()]
        print(f"Skipping scenario diff: missing {missing}")

    minuend_path = cm_data.mean_raster_results_path(
        results_version, REFERENCE_SCENARIO, LAST_FORECAST_YEAR, age_group_id, sex_id
    )
    # Historical years only run draw 0, so the last GBD year raster is the
    # per-draw file, not a coalesced mean.
    subtrahend_path = cm_data.raster_results_path(
        results_version, REFERENCE_SCENARIO, LAST_GBD_YEAR, age_group_id, sex_id, 0
    )
    if minuend_path.exists() and subtrahend_path.exists():
        diff = rt.load_raster(minuend_path) - rt.load_raster(subtrahend_path)
        plot_raster_diff(
            diff.to_numpy().astype(float) * 1000,
            # The label already holds a newline, so keep this on the same line
            # or the colorbar label runs off the figure edge.
            f"Difference in {label}, {LAST_FORECAST_YEAR} "
            f"({SCENARIO_DISPLAY_NAMES[REFERENCE_SCENARIO]}) - {LAST_GBD_YEAR}",
            results_dir / "raster_year_diff.png",
        )
    else:
        missing = [p.name for p in (minuend_path, subtrahend_path) if not p.exists()]
        print(f"Skipping year diff: missing {missing}")


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_cmip6_scenario()
@clio.with_age_group_id()
@clio.with_sex_id()
def coalesce_rasters_task(
    output_root: str,
    measure: str,
    results_version: str,
    cmip6_scenario: str,
    age_group_id: str,
    sex_id: str,
) -> None:
    """Coalesce one (scenario, age, sex)'s draw rasters into a mean raster."""
    [resolved_age_group_id] = clio.resolve_age_group_ids_for_measure(
        measure, [age_group_id]
    )
    coalesce_raster_draws_main(
        Path(output_root),
        measure,
        results_version,
        cmip6_scenario,
        clio.normalize_age_group_id(measure, resolved_age_group_id),
        int(sex_id),
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
def raster_diagnostics_task(
    output_root: str,
    measure: str,
    results_version: str,
) -> None:
    """Plot the raster diff diagnostics for a results version."""
    raster_diagnostics_main(Path(output_root), measure, results_version)
