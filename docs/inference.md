# Inference and Forecasting

Training fits on individual observations, each under its own climate and income
circumstances. Results, on the other hand, are prevalences and counts for every
location on the planet, over a century, under several scenarios. Bridging that
gap means we cannot simply call `predict` on the fitted model: we rebuild the
prediction pixel by pixel, from the coefficients and the gridded futures.

This page covers what the calculation does. For the commands, options and
outputs, see [Running the pipeline](running.md#2-inference).

The work is split across two steps:

* **Inference** produces, for one scenario/year/age/sex/draw, a prevalence
  raster and its aggregation to FHS locations.
* **Forecast** collects those pieces into scenario draw files and a summary
  table with populations and case counts.

## Building a prediction raster

For a given scenario, year, age group, sex and draw, inference accumulates the
linear predictor `z` over the grid, term by term, then applies the inverse
logit.

Each predictor contributes according to how the specification declared it:

* **Intercept** — read straight from the rasterised intercept saved at training
  time, so the fixed intercept and the location random intercept come in
  together.
* **Climate variables** — the scenario's climate raster for that year and draw,
  resampled onto the raster template and put through the *same* transformation
  object the model was fit with, then multiplied by its coefficient. Monthly
  climate variables are divided by 12 to match the units used in training.
* **Spline predictors** — the fitted curve saved as
  `{model}_{predictor}_spline_effects.parquet` is used as a lookup table: each
  pixel's covariate value is mapped to the nearest tabulated value's effect.
  The spline contributes that effect directly, with no separate coefficient.
* **Elevation** — from a static raster.
* **SDI** — from a rasterised SDI variable for that year.
* **Age, sex, `age_sex`** — a categorical coefficient, constant across the
  grid, so it is simply added to the accumulator. A value that has no
  coefficient is treated as the reference level.
* **Year** — evaluated at the maximum transformed year value rather than the
  year being predicted, i.e. the year effect is held at the end of the training
  period instead of being extrapolated.

Coefficients are loaded per `(age_group_id, sex_id)` through
`load_submodel_coefficients`, so a specification that fits age/sex submodels
and one that includes age and sex as categorical predictors both work.

## Income and the decile loop

Income is the one predictor that is not a single value per pixel. It arrives as
a distribution — ten population deciles per admin-2 unit — so a single pixel
holds ten different income levels.

Rather than predicting at mean income, we predict at each decile and average.
For each of the ten deciles, inference evaluates the full inverse logit with
that decile's income term added, and accumulates one tenth of the result. This
matters because the logistic link is non-linear: prevalence at mean income is
not the mean of prevalence over the income distribution, and for a skewed
income distribution the difference is not small.

The income scenario is tied to the climate scenario, since they come from the
same SSP narrative:

| CMIP6 scenario | Income scenario |
| --- | --- |
| `ssp126` | `better` |
| `ssp245` | `reference` (the reference scenario throughout the pipeline) |
| `ssp585` | `worse` |

Four combinations of income term are implemented, and they differ in what the
income coefficient is:

| Specification | Income term |
| --- | --- |
| Income linear, no threshold interaction | A single fitted coefficient |
| Income linear, interacted with a threshold flag | A per-pixel coefficient: the interaction coefficient scaled by the threshold variable's raster, plus the main income coefficient |
| Income as a spline, no interaction | Per-location effect read from the spline effects file, keyed on year, income scenario and decile |
| Income as a spline, interacted with a threshold flag | Both of the above |

Only a threshold-flag × income interaction is supported. Any other
`extra_terms` interaction raises `NotImplementedError` at inference time, even
though it trains without complaint.

## Aggregating to locations

A region's prevalence is the population-weighted mean of its pixels, so we need
a within-region population distribution. Inference multiplies the prevalence
raster by a population raster to get a count raster, then for each
most-detailed FHS shape takes the ratio of summed counts to summed population.

The population raster is a single high-resolution raster from the RRA
population model (`population-model/results/2026_05_16/wgs84_0p01/2023q1.tif`),
held **static across the whole estimation period**. It is used only to
distribute population *within* a region — the region's population *level* over
time comes from the FHS population forecasts in the next step. So the
assumption being made is that the within-region spatial distribution of
population does not change between now and 2100, not that population itself is
constant.

The raster template that everything is resampled onto is the 1 km Global Human
Settlement Layer template.

Only the location-aggregated table is kept for every task. Full prevalence
rasters are written for **2023 and 2100 only** — one raster per
scenario/year/age/sex/draw for a century would be unmanageable, and those two
years cover the "map of today vs map of the end of the century" diagnostic.

## Past years versus forecast years

Years before `FIRST_FORECAST_YEAR` (2024) are treated as past: they are
estimated once, for the reference scenario at draw 0, since the scenarios have
not diverged yet and the climate input is observed rather than projected. The
forecast step then reuses that single past-year file for every scenario, so
each scenario's series is continuous through history and only fans out from
2024 onward.

## Forecast: assembling scenarios

The forecast step reads the results specification to know exactly which files
to expect, then for each scenario pivots the per-task tables into draws-wide
form and writes `{scenario}.parquet`.

It then builds `forecast.parquet` from the draw means: prevalence joined to FHS
population (past and future, at draw level, future scenario 130), the implied
number of affected people, the difference in affected people against the
reference scenario, and the location/region/super-region names. A diagnostics
report is written alongside it as `forecast_diag.pdf`.

For `child_mortality` the age-specific mortality rates are combined across ages
into a single aggregate (`age_group_id` 1) by multiplying survival
probabilities, `1 - prod(1 - m_a)`, rather than summing rates.

## What inference does *not* do

The prevalences coming out of forecasting are the model's own, and they do not
line up with GBD. Reconciling them with GBD, and producing the final draws and
SEVs, is the [residual step](residual.md).
