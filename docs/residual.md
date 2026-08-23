# Residual step

The residual step runs after forecasting.  It takes the forecast draws written
by the inference/forecast step and reconciles them with GBD, then produces the
final prevalence (and, for the child growth failure measures, SEV) draws and the
diagnostic plots.

## Running it

```bash
# Launch on the cluster (one job)
strun residual -m stunting -r 2026_07_13.01

# Or run in the current process
sttask residual -m stunting -r 2026_07_13.01
```

Both need a results version that already has `forecast.parquet` and one
`{scenario}.parquet` per scenario in the results specification -- i.e. the
forecast step must have finished.  `strun residual` fails fast if
`forecast.parquet` is missing.

## Prerequisite: the cached GBD inputs

The residual step compares the forecast against GBD, so it reads cached GBD
estimates from `{output_root}/input/gbd_prevalence/`:

| File | Used for |
| --- | --- |
| `gbd_mean_{measure}_prevalence.parquet` | the residual model's outcome |
| `gbd_draws_{measure}_{metric}_100.parquet` | the intercept shift, for prevalence and SEV |
| `prev_to_sev_{measure}.parquet` | fitting the prevalence-to-SEV conversion |
| `age_group_metadata.parquet` | age group names on the diagnostic plots |

These are **not** created by the pipeline itself.  They come from
`src/rra_climate_health/data_prep/save_gbd_inputs.py`, which must be run
**with an IHME environment, not the project's pixi environment**.  Make
yourself a clone of the official IHME GBD environment (`gbdenv`) and use its
interpreter:

```bash
/path/to/your/gbdenv/bin/python \
    src/rra_climate_health/data_prep/save_gbd_inputs.py \
    --output-root /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition
```

That script is the only thing in the repository that imports the IHME database
libraries, and it is deliberately kept out of the `strun`/`sttask` CLI and out
of `pixi.lock`: pulling the IHME-internal dependency tree into the project
environment is exactly what we want to avoid.  For the same reason it imports
nothing from `rra_climate_health` (the package `__init__` imports `rpy2`, which
an IHME environment will not have) and depends only on `argparse`, `pandas` and
`numpy`.  Run it as a plain script path, not with `python -m`.

### Steps

`--measure` and `--steps` refresh a subset.  The steps are:

| Step | Writes | In the default set? |
| --- | --- | --- |
| `prevalence` | `gbd_{mean,draws}_{measure}_prevalence*` | yes |
| `mortality` | `gbd_{mean,draws}_{neonatal,child}_mortality_prevalence*` | yes |
| `age-metadata` | `age_group_metadata.parquet` | yes |
| `sev` | `gbd_{mean,draws}_{measure}_sev*` | no |
| `prev-to-sev` | `prev_to_sev_{measure}.parquet` | no |

Prevalence is pulled over the GBD hierarchy with
`ihme_cc_get_estimates.get_model_estimates` and population-weighted up to any
FHS most-detailed location GBD does not estimate directly.  LBW uses release 34
(GBD25); everything else uses release 16 (GBD23).  Mortality comes from the life
table (`life_table_parameter_id=3`, age groups 42 and 1), and since it has no
draws there, the draw files repeat the mean.

`sev` and `prev-to-sev` still need updating for the more recent changes to the
shared IHME functions, so they are not in the default step list.  They are
carried over as they last worked -- `get_draws(source="sev", ...)` and
`get_outputs`/`get_model_results` -- which is what produced the copies currently
on disk, but those entry points are not available in a current GBD environment.

If an input is missing, the residual step raises a `FileNotFoundError` naming
both the file and the command to create it.

## What it does

1. **Residual model** (`get_residual_prediction`).  For every location/age/sex,
   the residual is `GBD prevalence - reference-scenario model prevalence` over
   the years up to `LAST_GBD_YEAR`.  That residual is regressed on SDI with a
   super-region-specific slope and a location/age/sex intercept:

   ```
   residual_value ~ 0 + sdi + sdi*C(super_region_name) + C(location_age_sex)
   ```

   The fitted SDI slopes are then clamped so no super region has a positive
   slope (`adjust_sdi_slopes_non_positive`), and the model is used to predict a
   residual over the whole forecast horizon, which is added back onto the
   reference-scenario draws.  Set
   `ENFORCE_NON_POSITIVE_SDI_SLOPES = False` in `run_residual.py` to use the raw
   fitted coefficients instead.

2. **Intercept shift.**  The residual-adjusted draws are shifted by a constant
   (per location/age/sex/draw) so they hit the GBD draws exactly in
   `LAST_GBD_YEAR`.  Non-reference scenarios get the same shifted reference
   series plus that scenario's *unshifted* delta from the reference, so the
   scenario spread is preserved.

3. **Prevalence to SEV** (`convert_prev_to_sev`, only for `stunting`,
   `wasting`, `underweight`).  One mixed model of SEV on prevalence per age
   group, with a location random intercept, fit on the historical GBD
   prevalence/SEV pairs in `input/gbd_prevalence/prev_to_sev_{measure}.parquet`.
   The resulting SEV draws are logit-shifted onto GBD SEVs in `LAST_GBD_YEAR`.

4. **Diagnostics** (`residual_diagnostics.make_results_plots`).

## Outputs

All under `{output_root}/{measure}/results/{results_version}/`:

| File | Contents |
| --- | --- |
| `{scenario}_shifted.parquet` | Post-residual, post-shift prevalence draws, one file per scenario |
| `shifted_prevalence.parquet` | All scenarios concatenated |
| `adjusted_sev_draws.parquet` | Post-shift SEV draws (SEV measures only) |
| `sev_means.parquet` | Draw means of every intermediate and final quantity, used by the plots |
| `{measure}_prevalence_plots.pdf` | One page per location; sex-by-age grid of GBD / original model / residual model / shifted model prevalence |
| `{measure}_SEV_plots.pdf` | The same grid for SEVs |
| `superregion_prev.pdf` | Prevalence by scenario, globally and by super region |
| `table1_cumulative_case_counts.csv` | Cumulative case-count differences between scenario pairs at 2050 and 2100 |

## Preparing the FHS upload (on demand)

`src/rra_climate_health/residual/prepare_submission.py` reshapes the residual
step's `adjusted_sev_draws.parquet` into the per-scenario HDF5 files that the
FHS `save_results` upload takes as *its* input. It is a separate, on-demand
thing rather than a pipeline step, and is deliberately not registered on the
`strun`/`sttask` CLI:

```bash
python -m rra_climate_health.residual.prepare_submission \
    --measure stunting \
    --results-version 2026_07_13.01 \
    --submission-dir /ihme/scratch/users/<user>/save_results_fhs \
    --dry-run
```

`--submission-dir` is required -- in the source notebook this was hardcoded to
a personal scratch path. Underneath it the script creates
`{today}/{measure}/{fhs_scenario_id}/sev/sev_data_{fhs_scenario_id}.h5`, HDF key
`data`, one file per scenario, with scenarios mapped to FHS IDs
(`ssp126`->160, `ssp245`->159, `ssp585`->161) and the constant `measure_id` /
`metric_id` / `rei_id` / `release_id` columns stamped on.

Two things to know before running it for real:

* **It is big.** The upload wants a complete demographic grid, so the draws are
  reindexed onto every age group in `age_group_metadata.parquet` (152 of them),
  with zeros for the age groups the measure does not model. For the child growth
  failure measures that turns ~0.9M rows into ~34M rows, roughly 27 GB across
  the three scenario files. `--dry-run` reports the shape and the destination
  paths without writing.
* **Only years from `--year-upload-start` (default 2024) are included.**

## Where the code came from

Ported from the `malnutrition_fhs` repo:

| `malnutrition_fhs` | here |
| --- | --- |
| `src/residual.py` | `rra_climate_health/residual/run_residual.py` |
| `src/plotting.py` | `rra_climate_health/residual/residual_diagnostics.py` |
| `src/data_utils.py` | `rra_climate_health/residual/residual_data.py` |
| `src/constants.py` | `rra_climate_health/constants.py` |
| `src/2026_07_07_SaveGBDNew.ipynb` | `rra_climate_health/data_prep/save_gbd_inputs.py` |
| `src/PrepareSubmission2026.ipynb` | `rra_climate_health/residual/prepare_submission.py` |

See [residual_duplication_notes.md](residual_duplication_notes.md) for the
things that now exist in two places and what to do about them.
