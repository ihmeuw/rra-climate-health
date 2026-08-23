# Duplication introduced / found by the residual port

Bringing `malnutrition_fhs` in surfaced a handful of things that now exist in
more than one place, plus a few that were already duplicated inside this repo.
Nothing here is a bug today -- this is a list of decisions to make.

Legend: **[resolved]** = deduplicated as part of the port,
**[open]** = still duplicated, needs a call.

---

## 1. `load_population_timeseries` — **[resolved]**

* `malnutrition_fhs/src/data_utils.py::load_population_timeseries`
* `rra_climate_health/inference/run_inference.py::load_population_timeseries`

These were the same function; the only difference was that the inference copy
subset locations (`locs_of_interest`) and the `malnutrition_fhs` copy loaded all
of them.

**Done:** the residual step reuses the inference one, and its
`locs_of_interest` parameter now accepts `None` to mean "all locations".  The
two existing callers (`inference/admin2forecasting.py`,
`inference/admin2inference.py`) pass a location list and are unaffected.

**Still worth deciding:** it lives in `inference/run_inference.py`, which is a
slightly odd place for a shared helper.  Moving it to `utils.py` or a
`population.py` would be cleaner, but those two admin2 scripts import it as
`oldinf.load_population_timeseries`, so a move needs them updated too.

## 2. Population file paths — **[open, but not what it looks like]**

* `paths.py::FORECASTED_POPULATIONS_FILEPATH` → forecasting data `7`
* `inference/run_inference.py::FORECASTED_POPULATIONS_FILEPATH` → data `32`
  (plus `HISTORICAL_POPULATIONS_FILEPATH` → data `16`)
* `malnutrition_fhs/src/data_utils.py` → the same data `32` / `16` pair

This is not a live duplication: **the `paths.py` copy is never read.**  Nothing
in the repository accesses `paths.FORECASTED_POPULATIONS_FILEPATH`, and there is
no `getattr` on the module, so nothing is hiding.  The complete set of
`paths.<ATTR>` accesses in tracked files is `MODEL_ROOTS` (2),
`FHS_LOCATION_METADATA_FILEPATH` (2), `AGE_SPANS_FILEPATH` (1) and the three SDI
names this port added, plus one
`from rra_climate_health.paths import OUTPUT_ROOT` in a notebook.

The two things that look like uses are not:

* `run_inference.py` opens `FORECASTED_POPULATIONS_FILEPATH` at line ~545, but
  that resolves to its *own* module-level constant (data `32`, defined near the
  top of the file).  It does not import `paths` at all.
* `notebooks/2024_07_02_postprocessing.ipynb` assigns its own local variable of
  the same name, also pointing at data `7`.

So `paths.py` is not stale-and-shadowed, it is simply dead: the one place a
reader would naturally look for the population path holds a value nobody uses.
The same is true of `GLOBAL_POPULATION_FILEPATH`,
`LBD_ADMIN2_METADATA_FILEPATH`, `MODELS` and `RESULTS`, which are defined there
and referenced nowhere.

**Suggested:** delete `paths.py::FORECASTED_POPULATIONS_FILEPATH` -- that is a
runtime no-op.  Do *not* "move the live paths into `paths.py`": that would
create a coupling that does not exist today and would risk changing what
inference reads.  `paths.py` is intentionally left untouched by this port apart
from the SDI additions.

## 3. Constants — **[partly resolved]**

New `rra_climate_health/constants.py` holds what was in
`malnutrition_fhs/src/constants.py`.  As part of that,
`inference/run_inference.py` now imports `AGE_GROUP_AGGREGATES`,
`FIRST_FORECAST_YEAR` and `REFERENCE_SCENARIO` from it instead of defining its
own identical copies (the names are still bound at module level there, so
anything importing them from `run_inference` keeps working).

**[open]** Age groups by measure exist twice, with genuinely different values:

* `cli_options.py::AGE_GROUP_IDS_BY_MEASURE` — strings, the age groups models
  are *fit* on; `anemia` is `8..14`, `child_mortality` is `age_1_m ... age_60_m`.
* `constants.py::VALID_AGE_GROUPS_FOR_MEASURE` — ints, the age groups results
  are *reported* for; `anemia` includes `34, 238, 389, 15`,
  `child_mortality` is `[1]`, and there is no `lbw` entry.

The residual diagnostics use the second one to decide which population to pull.
These are two different concepts sharing a shape, so my read is that they should
stay separate but be renamed to say which is which (e.g.
`MODELED_AGE_GROUPS_BY_MEASURE` vs `REPORTED_AGE_GROUPS_BY_MEASURE`) and live in
the same file.  The rename is still open; the `lbw` gap has since been filled.

**[resolved]** The *typing* half of this is now settled — the part that actually
broke runs.  `AGE_GROUP_IDS_BY_MEASURE` still holds strings (click choices are
strings), but `cli_options.normalize_age_group_id` converts them to int at the single
point where tasks are dispatched, so results tables, scenario parquets and
`results_spec.yaml` all carry int `age_group_id` again.  Only measures listed in
`cli_options.NAMED_AGE_STRATA_MEASURES` keep strings, and only until the forecast step
collapses them.  `ResultsSpecification` coerces numeric age groups to int on read, so
specs written during the string-typed window heal themselves, and
`load_population_timeseries` now raises `TypeError` rather than silently producing an
all-NaN population join.

## 4. Location hierarchy loaders — **[resolved]**

* `malnutrition_fhs`'s `get_fhs_location_metadata()` reads
  `input/fhs_location_metadata.parquet`
* `ClimateMalnutritionData.load_fhs_hierarchy()` reads
  `input/fhs_hierarchy.parquet`
* `data_prep/location_mapping.py::FHS_HIERARCHY_PATH` and
  `paths.py::FHS_LOCATION_METADATA_FILEPATH` point at the two files separately

The two parquets are byte-for-byte identical today (verified: same 513 rows, same
31 columns, `DataFrame.equals` is `True`).  The port uses
`load_fhs_hierarchy()`.

**Still worth deciding:** two identical files in `input/` is a trap — they will
drift.  Recommend keeping one and making the other a symlink, or deleting
`fhs_location_metadata.parquet` and repointing `paths.py`.

## 5. Age metadata — **[open]**

Two different files, both in use:

* `input/gbd_prevalence/age_group_metadata.parquet` (152 rows) — used by
  `malnutrition_fhs`'s plots, so now by `residual_diagnostics.py` via the new
  `ClimateMalnutritionData.load_age_group_metadata()`
* `input/gbd_prevalence/age_metadata.parquet` — read directly by
  `inference/inference_diagnostics.py::plot_gbd_comparison`

Not the same file, so not strictly duplication, but two age-metadata sources in
one pipeline is worth a look.

## 6. Cumulative-difference tables — **[open]**

* `inference/inference_diagnostics.py::get_cumulative_differences` — works off
  the means in `forecast.parquet`, reports `delta` vs the reference scenario,
  renders into the `forecast_diag.pdf` reportlab doc.
* `residual/residual_diagnostics.py::get_cumulative_count_scenario_differences_table`
  — works off the post-residual draws, reports differences between *pairs* of
  scenarios with 95% UIs, renders to CSV.

Same idea, different inputs and different outputs, so I kept both.  If the
post-residual numbers are the ones people actually quote, the inference-side
table is arguably now redundant.

## 7. `save_gbd_inputs.py` is deliberately standalone — **[open by design]**

`data_prep/save_gbd_inputs.py` re-declares `ME_ID_DICT`, `REI_ID_DICT`,
`resample_like_fhs` and `aggregate_forecast_hierarchy`, all of which also exist
in `residual/residual_data.py`.

This one is intentional and should stay.  The script runs under a clone of the
official IHME GBD environment to reach the database libraries, and it cannot
import from `rra_climate_health` at all because the package `__init__` imports
`rpy2`.  Sharing code would mean either pulling the IHME dependency tree into
`pixi.lock` or making the package `__init__` importable without `rpy2` --
neither is worth it here.

The GBD ID dicts live *only* in the script (they were briefly in `constants.py`,
but nothing in the package used them).  `resample_like_fhs` and
`aggregate_forecast_hierarchy` are genuinely in two places; the copies carry
comments pointing at each other.  `aggregate_forecast_hierarchy` is the larger
of the two (~80 lines of pandas) and is the one most worth revisiting if the
`rpy2`-free-`__init__` question is ever reopened -- the script needs it because
GBD does not estimate every FHS most-detailed location, so prevalence has to be
population-weighted up the GBD hierarchy before it can be subset to FHS.

## 8. Hierarchy aggregation — **[open, low priority]**

`residual_data.py::aggregate_forecast_hierarchy` /
`aggregate_age_and_sex` do population-weighted aggregation by exploding
`path_to_top_parent`.  `inference/run_inference.py::forecast_scenarios` does a
lighter-weight version of the same thing inline (merge population, multiply,
sum).  Not worth unifying unless the aggregation logic starts to diverge.

---

## Deviations from the `malnutrition_fhs` source

Deliberate, so worth listing:

* **Paths.**  Every hardcoded
  `/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/...`
  f-string is now a `ClimateMalnutritionData` method, so the step honours
  `--output-root`.  New methods: `load_scenario_draws`,
  `save/load_shifted_scenario_draws`, `save/load_shifted_prevalence`,
  `save/load_adjusted_sev_draws`, `save/load_sev_means`,
  `load_age_group_metadata`, and the `gbd_inputs` property.
* **`assert` → `raise`.**  The asserts in `convert_prev_to_sev`,
  `aggregate_forecast_hierarchy`, `aggregate_age_and_sex` and
  `resample_like_fhs` now raise `ValueError` with a message naming the offending
  locations/age groups.  Same conditions, different exception type.
* **The database fallbacks are gone.**  `data_utils.py` had `get_gbd_data_db`
  and `get_prev_to_sev_table_db`, which hit `get_draws`/`db_queries` whenever a
  cached parquet was missing.  Those are not ported: the package must not
  depend on the IHME-internal stack.  Creating those inputs is now the explicit
  job of `data_prep/save_gbd_inputs.py` (ported from
  `malnutrition_fhs/src/2026_07_07_SaveGBDNew.ipynb`, which is what actually
  produced them), and the loaders raise a `FileNotFoundError` naming the file
  and the command to create it.

  Worth knowing: the old `get_prev_to_sev_table_db` fallback used `release_id=9`
  and applied no year filter, while the notebook uses `release_id=16` and
  filters to 1990-2022 because `get_model_results` ignores its `year_id`
  argument.  The script follows the notebook.
* **`sev` and `prev-to-sev` are not in the script's default steps.**  Both
  still need updating for the more recent changes to the shared IHME functions,
  so they are carried over as they last worked (`get_draws(source="sev", ...)`
  and `get_outputs`/`get_model_results`) rather than rewritten against APIs that
  cannot express them -- SEVs are keyed by `rei_id`, which
  `ihme_cc_get_estimates.get_model_estimates` has no parameter for.  The cached
  copies on disk are what the residual step currently reads.
* **`plot_multiple_superregion_prevalence_rate` population leak.**  In the
  original, the GBD loop reused the `population` variable left over from the
  last iteration of the forecast loop.  For a single measure — which is all the
  pipeline ever calls it with — that is the same object, so behaviour is
  unchanged; the port caches population per measure so the multi-measure case is
  also right.  It still passes the full population to the GBD aggregation and
  the draw-column subset to the forecast aggregation, as the original did.
* **Query-string hoisting.**  `"location_id in @all_prev.index.get_level_values(...)"`
  inside a `DataFrame.query` string is now hoisted to a local (`modeled_locs`).
* **`prepare_submission.py` drops a redundant `reset_index()`.**  The notebook
  called `reset_index()` twice in a row; the second one ran on an already-reset
  RangeIndex, so it added a spurious `index` column to the uploaded frame.  The
  data is otherwise identical.  It also re-declares `REI_ID_DICT`, for the same
  reason as the GBD input script, and it validates rather than eyeballs the two
  checks the notebook did by hand (unmapped scenarios and leftover NaNs now
  raise).
* **Dropped `make_paper_prevalence_plot`.**  It called `get_population()`, which
  was commented out in `data_utils.py`, so it could not run.  Say the word and I
  will bring it back on top of `load_population_timeseries`.
* **Dropped a no-op.**  `table_df['formatted'].str.replace('–', '–')` in the
  Lancet formatting replaced a character with itself.
* **`resample_like_FHS` → `resample_like_fhs`** to match the repo's naming.
* **Plot styles and label maps** are module-level constants rather than dicts
  rebuilt on every call.  Same content.
* **Dependencies.**  `statsmodels` and `patsy` are now declared in
  `[tool.pixi.dependencies]`.  Both were already in `pixi.lock`, but only
  transitively -- `statsmodels` via the `seaborn` meta-package and `patsy` via
  `statsmodels` -- and the residual step imports them directly.  `tqdm` is left
  undeclared because `rra-tools` requires it, so it is already a first-class
  entry in the lock.  No IHME-internal package is added.
