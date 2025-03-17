from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, Field, model_validator


class ScalingStrategy(StrEnum):
    IDENTITY = "identity"
    MIN_MAX = "min_max"
    STANDARDIZE = "standardize"
    INNER_NINETY_FIVE = "inner_ninety_five"


class ScalingSpecification(BaseModel):
    type: Literal["scaling"] = "scaling"
    strategy: ScalingStrategy = ScalingStrategy.IDENTITY


class BinningCategory(StrEnum):
    HOUSEHOLD = "household"
    LOCATION = "location"
    COUNTRY = "country"


BINNING_CATEGORY_GROUPBY = {
    BinningCategory.HOUSEHOLD: ["nid", "hh_id", "psu", "year_start"],
    BinningCategory.LOCATION: ["lat", "long"],
    BinningCategory.COUNTRY: ["iso3"],
}


class BinningStrategy(StrEnum):
    QUANTILES = "quantiles"
    EQUAL = "equal"
    READABLE_5 = "readable_5"
    ZERO_ONE_MORE = "0_1_more"
    ZERO_MORE = "0_more"
    ZERO_MORE_READABLE = "0_more_readable"
    CUSTOM_DAYSOVER = "custom_daysover"


class BinningSpecification(BaseModel):
    type: Literal["binning"] = "binning"
    strategy: BinningStrategy
    category: BinningCategory
    nbins: int = Field(10, gt=0)

    @property
    def groupby_columns(self) -> list[str]:
        return BINNING_CATEGORY_GROUPBY[self.category]


class MaskingSpecification(BaseModel):
    type: Literal["masking"] = "masking"
    from_column: str
    threshold: float


class CategoricalSpecification(BaseModel):
    type: Literal["categorical"] = "categorical"


TransformSpecification: TypeAlias = (
    BinningSpecification
    | ScalingSpecification
    | MaskingSpecification
    | CategoricalSpecification
)


class OutcomeVariable(StrEnum):
    WASTING = "wasting"
    STUNTING = "stunting"
    UNDERWEIGHT = "underweight"
    LOW_BMI = "low_bmi"
    ANEMIA = "anemia"


class PredictorSpecification(BaseModel):
    name: str = "intercept"
    transform: TransformSpecification = Field(
        ScalingSpecification(),
        discriminator="type",
    )
    random_effect: str = ""
    version: str = ""

    @property
    def raw_variables(self) -> list[str]:
        if hasattr(self.transform, "from_column"):
            variables = [self.transform.from_column]
        else:
            variables = [self.name]
        if self.random_effect:
            variables.append(self.random_effect)
        if self.transform.type == "binning":
            variables += self.transform.groupby_columns
        return variables


class GridSpecification(BaseModel):
    x: PredictorSpecification
    y: PredictorSpecification
    name: str = "grid_cell"
    random_effect: str = ""

    @model_validator(mode="after")  # type: ignore[arg-type]
    def check_no_predictor_random_effects(
        cls,  # noqa: N805
        v: "GridSpecification",
    ) -> "GridSpecification":
        if v.x.random_effect:
            msg = "Grid predictor x cannot have a random effect"
            raise ValueError(msg)
        if v.y.random_effect:
            msg = "Grid predictor y cannot have a random effect"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")  # type: ignore[arg-type]
    def check_transform_is_binning(cls, v: "GridSpecification") -> "GridSpecification":  # noqa: N805
        if v.x.transform.type != "binning":
            msg = "Grid predictor x must be binned"
            raise ValueError(msg)
        if v.y.transform.type != "binning":
            msg = "Grid predictor y must be binned"
            raise ValueError(msg)
        return v

    @property
    def raw_variables(self) -> list[str]:
        return list(set(self.x.raw_variables + self.y.raw_variables))

    @property
    def transform_map(self) -> dict[str, TransformSpecification]:
        return {
            self.x.name: self.x.transform,
            self.y.name: self.y.transform,
        }

    @property
    def grid_spec(self) -> dict[str, list[str]]:
        return {"grid_order": [self.x.name, self.y.name]}


class HoldoutType(StrEnum):
    random = "random"
    no_holdout = "no_holdout"


class HoldoutSpecification(BaseModel):
    type: HoldoutType = HoldoutType.random
    proportion: float = 0.2
    seed: int = 42


class SubmodelSpecification(BaseModel):
    name: str


class VersionSpecification(BaseModel):
    training_data: str
    model: str | None = None
    description: str | None = None


class ModelSpecification(BaseModel):
    version: VersionSpecification
    measure: OutcomeVariable
    holdout: HoldoutSpecification = HoldoutSpecification()
    submodel_vars: list[SubmodelSpecification] | None = None
    predictors: list[PredictorSpecification] = Field(default_factory=list)
    grid_predictors: GridSpecification | None = None
    extra_terms: list[str] | None = None

    @property
    def random_effects(self) -> list[str]:
        random_effects = [predictor.random_effect for predictor in self.predictors]
        if self.grid_predictors:
            random_effects.append(self.grid_predictors.random_effect)
        return [re for re in random_effects if re]

    @property
    def raw_variables(self) -> list[str]:
        variables: list[str] = [self.measure]
        for predictor in self.predictors:
            variables += predictor.raw_variables

        if self.grid_predictors:
            variables += self.grid_predictors.raw_variables

        variables = list(set(variables))

        return variables

    @property
    def transform_map(self) -> dict[str, TransformSpecification]:
        transform_map = {
            predictor.name: predictor.transform for predictor in self.predictors
        }
        if self.grid_predictors:
            transform_map.update(self.grid_predictors.transform_map)
        return transform_map

    @property
    def lmer_formula(self) -> str:
        formula = f"{self.measure.value} ~"

        predictors: list[PredictorSpecification] = []

        if self.grid_predictors:
            grid_cell_predictor = PredictorSpecification(
                name=self.grid_predictors.name,
                transform=CategoricalSpecification(),
                random_effect=self.grid_predictors.random_effect,
            )
            predictors.append(grid_cell_predictor)

        predictors += self.predictors
        random_effects: dict[str, list[str]] = {}
        for predictor in predictors:
            predictor_repr = "1" if predictor.name == "intercept" else predictor.name
            predictor_repr = (
                f"C({predictor_repr})"
                if predictor.transform.type == "categorical"
                else predictor_repr
            )
            if predictor.random_effect:
                if predictor.random_effect in random_effects:
                    random_effects[predictor.random_effect].append(predictor_repr)
                else:
                    random_effects[predictor.random_effect] = [predictor_repr]
                formula += f" {predictor_repr}  +"
            else:
                formula += f" {predictor_repr} +"
        for random_effect, variables in random_effects.items():
            formula += f" ({' + '.join(variables)} | {random_effect}) +"
        if self.extra_terms:
            for term in self.extra_terms:
                formula += f" {term} +"
        formula = formula.rstrip(" +")
        return formula

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ModelSpecification":
        with Path(yaml_path).open("r") as f:
            yaml_dict = yaml.safe_load(f)
        return cls.model_validate(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
