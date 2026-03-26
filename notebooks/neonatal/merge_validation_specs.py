import pandas as pd
import os
from pathlib import Path
import yaml
import re
from rra_climate_health.data import DEFAULT_ROOT

output_root = Path(DEFAULT_ROOT)
RESULTS_ROOT = output_root / "neonatal_mortality" / "models"
validation_results = pd.read_csv(Path(RESULTS_ROOT) / "validation_results.csv")


def build_formula_from_spec(spec: dict) -> str:
    model_type = spec.get("model_type", "unknown")
    measure = spec.get("measure", "y")
    consumption_terms = []
    other_spline_terms = []
    plain_terms = []
    re_terms = []
    for pred in spec.get("predictors", []):
        name = pred["name"]
        spline = pred.get("spline")
        random_effect = pred.get("random_effect", "")

        if random_effect:
            re_terms.append(f's({random_effect}, bs="re")')
        elif spline:
            bs = spline.get("bs", "")
            k = spline.get("k", "")
            knot_strategy = spline.get("knot_strategy", "")
            term = f's({name}, bs="{bs}", k={k}, knot_strategy="{knot_strategy}")'
            if name == "consumption_pd":
                consumption_terms.append(term)
            else:
                other_spline_terms.append(term)
        else:
            plain_terms.append(name)

    ordered = consumption_terms + other_spline_terms + plain_terms + re_terms
    rhs = " + ".join(ordered)
    return f"{model_type}({measure} ~ {rhs})"


specs = []
for mv in validation_results["model_version"]:
    spec_path = RESULTS_ROOT / mv / "specification.yaml"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        specs.append(build_formula_from_spec(spec))
    else:
        specs.append(None)

validation_results["specification"] = specs

# save out
validation_results.sort_values("gbd_rmse", inplace=True)
validation_results.to_csv(
    Path(RESULTS_ROOT) / "validation_results_with_specs.csv", index=False
)
