# Current Versions

The model version and results version currently considered current for each
measure. A results version is the useful thing to quote — it pins the model
version, which pins the specification and the training data version (see
[Versions and how they chain](running.md#versions-and-how-they-chain)).

All paths below are relative to the output root:

```
/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/{measure}/
```

!!! warning "Fill this in"
    The versions in the table below were read off the filesystem on
    **2026-08-21** by picking, for each measure, the most recent results
    version that has a complete set of outputs. **They have not been confirmed
    as the ones we actually want to quote** — replace them with the blessed
    versions and delete this admonition.

## Current versions by measure

| Measure | Model version | Results version | Notes |
| --- | --- | --- | --- | --- |
| `stunting` | `2025_04_03.01` | `2025_05_09.01` | Paper submission | 
| `wasting` | `2025_04_03.01` | `2025_05_09.01` | Paper submission | 
| `underweight` | `2025_04_03.01` | `2025_05_09.01` | Paper submission | 
| `stunting` | `2026_07_06.04` | `2026_07_13.01` | Latest stunting | 
| `wasting` | `2026_07_21.01` | `2026_07_26.01` | Latest wasting | 
| `underweight` | `2026_07_06.03` | `2026_07_07.03` | Latest underweight | 
| `anemia` | `2026_08_01.32` | `2026_08_01.02` | Latest anemia | 
| `lbw` | `2026_08_18.01` | `2026_08_18.01` |  Scenario draw files exist but no `forecast.parquet`; residual step not run |
| `neonatal_mortality` | `2026_08_03.02` | `2026_08_03.04` | Candidate run, without precipitation |
| `neonatal_mortality` |  `2026_08_03.01`|`2026_08_03.03` | Latest neonatal |
| `child_mortality` | `2026_08_17.01` | `2026_08_17.07` | Bad run |


## What "complete" means

A results version has been through the whole pipeline when the directory
`results/{results_version}/` contains:

| File | Written by | Applies to |
| --- | --- | --- |
| `results_spec.yaml` | `strun inference` | all measures |
| `{scenario}.parquet` (one per scenario) | `forecast` | all measures |
| `forecast.parquet` | `forecast` | all measures |
| `shifted_prevalence.parquet` | `residual` | all measures |
| `adjusted_sev_draws.parquet` | `residual` | `stunting`, `wasting`, `underweight` only |

Only the child growth failure measures get SEVs, so a missing
`adjusted_sev_draws.parquet` for `anemia`, `lbw` or the mortality measures is
expected rather than a sign of an incomplete run.

## Keeping this page current

When a run becomes the one to quote, update its row here. The three things
worth recording are the measure, the results version, and one line on what
changed — the rest is recoverable from the version directories themselves:

```sh
ROOT=/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition
MEASURE=stunting
RESULTS=2026_07_13.01

# The model version behind this results version, plus scenarios/years/draws
cat $ROOT/$MEASURE/results/$RESULTS/results_spec.yaml

# The specification that model was fit with, and its training data version
MODEL=$(grep -Po '(?<=model: ).*' $ROOT/$MEASURE/results/$RESULTS/results_spec.yaml)
cat $ROOT/$MEASURE/models/$MODEL/specification.yaml
```

