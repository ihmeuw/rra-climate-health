# Training Methodology

The objective is to estimate the effect of climate variables, income and other
covariates on a health outcome, at the level of an individual observation, so
that the fitted relationship can then be applied to gridded climate and income
futures. For how to actually launch a fit, see
[Running the pipeline](running.md#1-training).

Everything about a fit is declared in a **model specification** YAML file —
the outcome, the training data version, the predictors, the transformations
applied to each, the random effects, and whether the model is fit as a
generalised linear mixed model or as a shape-constrained additive model. The
specification is the unit of reproducibility: `strun training` copies it into
the model version directory as `specification.yaml`, so the version records
exactly what was fit.

The examples in `specifications/` are the best starting point;
`model_specification.py` is the authoritative schema.

## Model types

`model_type` selects the estimator. Both fit the outcome on the logit scale
with a binomial family, and both support a location random intercept.

| `model_type` | Estimator | Notes |
| --- | --- | --- |
| `lmer` (default) | `pymer4.models.Lmer`, `family="binomial"` | Mixed-effects logistic regression. Random slopes as well as intercepts are expressible in the formula; the fit fails loudly if `lme4` emits convergence warnings. |
| `scam` | R's `scam` package, `binomial(link="logit")` | Shape-constrained additive model. Lets a predictor enter as a monotone spline instead of a single linear coefficient. Random effects are fit as `s(var, bs="re")` and one random effect only. |

The formula is not written by hand. It is assembled from the predictor list by
`ModelSpecification.lmer_formula` — categorical predictors become `C(var)`,
spline predictors become `s(var, bs=..., k=...)`, random effects are collected
per grouping variable, and anything in `extra_terms` is appended verbatim.
Printing that property is the quickest way to check that a specification says
what you meant:

```python
from rra_climate_health.model_specification import ModelSpecification
print(ModelSpecification.from_yaml("specifications/stunting.yaml").lmer_formula)
```

## Outcomes

`measure` is one of `stunting`, `wasting`, `underweight`, `low_bmi`, `anemia`,
`lbw`, `neonatal_mortality`, `child_mortality`. It selects both the outcome
column in the training data and the measure directory that outputs are written
under.

## Predictors

Each entry in `predictors` is a name, a transformation, and optionally a random
effect, an input data version, and a spline:

```yaml
predictors:
  - name: "intercept"
    random_effect: "ihme_loc_id"
  - name: "age_group_id"
    transform:
      type: "categorical"
  - name: "days_over_30C"
    transform:
      type: "scaling"
      strategy: "inner_ninety_five"
      scale_source: "data"
    spline:
      bs: "mpi"
      k: 4
  - name: "ldi_pc_pd"
    version: "v5"
    transform:
      type: "scaling"
      strategy: "inner_ninety_five"
      scale_source: "inference"
    spline:
      bs: "mpd"
      k: 4
```

What is available:

* **Any of the rasterised climate variables**, since inference has to be able
  to reproduce the predictor from a raster. Adding a new rasterised variable
  takes a few code changes; a variable that varies geographically but is not
  rasterised has to be rasterised first.
* **Income** as `ldi_pc_pd` (LDI per capita per day) or a `consumption_pd`
  variable. `version` pins which income version to use.
* **`elevation`**, read from a static raster rather than the training data.
* **Age and sex** as `age_group_id`, `sex_id` or the combined `age_sex`, either
  as categorical predictors *or* as `submodel_vars` (below).
* **Other covariates**: `sdi`, and a year variable (`year_start`, `year`,
  `year_id` or `birth_year`).
* **Random intercepts by location**, either country level (`ihme_loc_id`) or
  FHS admin-2 (`lbd_admin2_id`). Inference supports location random
  *intercepts* only.

### Transformations

Every predictor declares how the raw column is turned into a model covariate.
The fitted transformer is saved with the model so that inference applies the
identical transformation to the rasters.

**`scaling`** — `strategy` is one of:

| Strategy | Effect |
| --- | --- |
| `identity` | Pass through unchanged |
| `min_max` | Scale to [0, 1] |
| `standardize` | Centre and scale to unit variance |
| `inner_ninety_five` | Scale so the 2.5th and 97.5th percentiles land at 0 and 1, leaving the tails outside that range |

`scale_source` decides *which distribution* the scaler is fit on, and it
matters more than it looks:

* `data` (default) fits on the training data.
* `inference` fits on the distribution the predictor will have at prediction
  time — the admin-2 LDI distributions for income, or the climate aggregates
  for a climate variable — so that a covariate value means the same thing in
  training and in inference. Only `inner_ninety_five` is supported here, and
  the predictor's `version` selects which input version to fit against.

**`binning`** — discretise into `nbins` bins. `strategy` is one of
`quantiles`, `equal`, `readable_5`, `0_1_more`, `0_more`, `0_more_readable`,
`custom_daysover`; `category` says what the quantiles are taken *within*:
`household` (`nid`/`hh_id`/`psu`/`year_start`), `location` (`lat`/`long`) or
`country` (`iso3`).

**`masking`** — turn a continuous column into a 0/1 flag via
`from_column` and `threshold`. This is how threshold predictors such as
"was this observation exposed to any days over 30 °C" are built. Note the name
of the predictor is the *new* flag; `from_column` is the source.

**`categorical`** — treat as a factor, entering the formula as `C(var)`.

### Splines (`scam` only)

A predictor with a `spline` block enters as a smooth rather than a single
coefficient:

```yaml
spline:
  bs: "mpi"            # scam basis: mpi = monotone increasing, mpd = monotone decreasing
  k: 4                 # basis dimension
  knot_strategy: "quantiles"
```

The monotone bases are the point of using `scam`: they let us assert that, say,
prevalence is non-increasing in income without also asserting that the
relationship is linear.

`knot_strategy` places the `k - 4` inner knots, and is optional — omit it to
let `scam` choose:

| Strategy | Inner knots at |
| --- | --- |
| `quantiles` | Equally spaced quantiles of the covariate, falling back to quantiles of its unique values if that produces duplicate knots |
| `quantile_unique` | Equally spaced quantiles of the unique values |
| `equal` | Equally spaced between the covariate's min and max |
| `harrell` | Harrell's recommended quantiles for the given number of knots (defined for 3–7 inner knots) |
| `custom_knots` | The values given in `knots:`, put through the predictor's own transformation first |

The knot vector is then extended three knots beyond each end of the data at the
mean inner-knot spacing, as `scam` requires.

For each spline predictor, training saves the fitted curve to
`{model}_{predictor}_spline_effects.parquet`. Inference reads that curve and
looks values up in it, rather than re-evaluating the basis — so the spline
effects file is a required input to inference, not just a diagnostic.

## Interactions, grids, and submodels

**`extra_terms`** are appended to the formula as written, which is how
interactions are expressed:

```yaml
extra_terms:
  - "any_days_over_30C * ldi_pc_pd"
```

Inference implements the threshold-flag × income interaction specifically, and
raises `NotImplementedError` for anything else, so an arbitrary extra term will
train but not predict.

**`grid_predictors`** cross two binned predictors into a single categorical
`grid_cell`, optionally with a random effect on it — a way to let a
two-dimensional surface be estimated non-parametrically. Both `x` and `y` must
use a `binning` transform and neither may carry its own random effect.

**`submodel_vars`** fits *separate models* per value instead of adding a term:

```yaml
submodel_vars:
  - name: "age_group_id"
  - name: "sex_id"
```

`strun training` takes the cross product of those variables' observed values in
the training data and submits one job per combination, each writing its own
pickle and coefficient files into the same model version. This is the
alternative to treating age and sex as categorical predictors: more flexible,
since every coefficient is free to differ by age and sex, at the cost of
fitting on less data per model and losing the shared-information pooling.

Inference loads coefficients through `load_submodel_coefficients` keyed on
`(age_group_id, sex_id)`, so either arrangement predicts.

**`holdout`** is `no_holdout`, or `random` with a `proportion` and a `seed`.

## What training produces

Training loads the training data version named in the specification, applies
each predictor's transformation, drops the submodel subset it was asked for,
fits, and then saves the model, its coefficients, and its diagnostics. For the
full output listing see
[Running the pipeline](running.md#1-training).

Two outputs are worth calling out because inference depends on them:

* The **rasterised intercept** — the fixed intercept plus the location random
  intercept, burned onto the raster template. It is written once per model
  version, for the full model rather than per submodel, so that inference does
  not have to rasterise location effects on every task.
* The **spline effects** files described above.

Null values in the raw predictor columns are reported but not dropped, so a
row count that looks low in the training log is worth chasing down.
