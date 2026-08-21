import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, precision_recall_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from pymer4.models import Lmer
from dataclasses import dataclass
from typing import List, Optional, Tuple
from rpy2.robjects import pandas2ri, packages,  ListVector, FloatVector

import pathlib

from rra_climate_health.model_specification import ModelType, SplineSpecification
from rra_climate_health.training.training_diagnostics import merge_gbd_data

@dataclass
class ValidationConfig:
    """Configuration for model validation splits."""
    n_splits: int = 2
    test_size_locations: float = 0.2
    test_size_recent_years: float = 0.15
    random_seed: int = 42

def prepare_cv_splits(df: pd.DataFrame, 
                      location_col: str, 
                      year_col: str, 
                      config: ValidationConfig):
    """
    Generates K-Fold CV splits. In each fold:
    1. A subset of locations is held out (Unseen Locations).
    2. For the training locations, the most recent years are held out (Unseen Years).
    """
    all_locations = df[location_col].unique()
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    
    for train_loc_idx, test_loc_idx in kf.split(all_locations):
        seen_locations = all_locations[train_loc_idx]
        unseen_locations = all_locations[test_loc_idx]
        
        # Scenario A: Entirely unseen locations
        test_unseen_loc_df = df[df[location_col].isin(unseen_locations)]
        
        # Scenario B: Unseen years for seen locations
        seen_df = df[df[location_col].isin(seen_locations)]
        train_indices = []
        test_unseen_year_indices = []
        
        for loc in seen_locations:
            loc_data = seen_df[seen_df[location_col] == loc]
            years = sorted(loc_data[year_col].unique())
            
            if len(years) <= 1:
                train_indices.extend(loc_data.index.tolist())
            else:
                n_test_years = max(1, int(len(years) * config.test_size_recent_years))
                train_years = years[:-n_test_years]
                test_years = years[-n_test_years:]
                
                train_indices.extend(loc_data[loc_data[year_col].isin(train_years)].index.tolist())
                test_unseen_year_indices.extend(loc_data[loc_data[year_col].isin(test_years)].index.tolist())
        
        yield (
            df.loc[train_indices], 
            test_unseen_loc_df, 
            df.loc[test_unseen_year_indices]
        )

def calculate_metrics(y_true, y_prob):
    """Utility to calculate various performance metrics."""
    # Handle cases with very little data to avoid errors
    if len(np.unique(y_true)) < 2:
        return {'log_loss': np.nan, 'brier_score': np.nan, 'ece': np.nan, 'auc_pr': np.nan}

    metrics = {}
    metrics['log_loss'] = log_loss(y_true, y_prob, labels=[0, 1])
    metrics['brier_score'] = np.mean((y_true - y_prob) ** 2)
    
    # Calibration Error (ECE approximation)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    metrics['ece'] = np.mean(np.abs(prob_true - prob_pred))
    
    # AUC-PR (Better than AUC-ROC for imbalanced classes)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics['auc_pr'] = auc(recall, precision)
    
    return metrics

def validate_model(df: pd.DataFrame, 
                   model_spec, 
                   target_measure: str, 
                   year_variable: str,
                   var_info: dict,
                   config: Optional[ValidationConfig] = None):
    """
    Validated model performance using K-Fold CV with custom spatio-temporal splits.
    """
    config = config or ValidationConfig()
    location_variable = model_spec.random_effects[0]
    
    all_fold_results = []

    for fold, (train_df, test_loc, test_year) in enumerate(prepare_cv_splits(df, location_variable, year_variable, config)):
        print(f"Processing Fold {fold+1}/{config.n_splits}...")
        
        # Fit Model
        model_type = model_spec.model_type
        if model_type == ModelType.LINEAR_MIXED_EFFECTS:
            model = Lmer(model_spec.lmer_formula, data=train_df, family="binomial")
            model.fit(summarize=False)
        elif model_type == ModelType.SPLINE_MIXED_EFFECTS:
            pandas2ri.activate()
            scam_lib = packages.importr('scam')
            base = packages.importr('base')
            stats = packages.importr('stats')
            train_levels = train_df['ihme_loc_id'].cat.categories

            knots_dict = {}
            for predictor in model_spec.predictors:
                if predictor.spline is not None and predictor.spline.knot_strategy is not None:
                    knots = get_knot_values(df, predictor.name, predictor.spline, var_info)
                    print(f"Knots for {predictor.name}: {knots}")
                    knots_dict[predictor.name] = FloatVector(knots)
            knots = ListVector(knots_dict) if len(knots_dict) > 0 else None
            if knots is not None:
                model = scam_lib.scam(stats.as_formula(model_spec.lmer_formula), data=df, 
                                    family = stats.binomial(link = "logit"), knots = knots )
            else:
                model = scam_lib.scam(stats.as_formula(model_spec.lmer_formula), data=df, family = stats.binomial(link = "logit") )

        evaluation_sets = [
            ('unseen_locations', test_loc),
            ('unseen_years', test_year)
        ]

        test_sets = []
        for label, test_set in evaluation_sets:
            if test_set.empty: continue

            if model_type == ModelType.LINEAR_MIXED_EFFECTS:
                y_pred = model.predict(test_set, verify_predictions=False, use_rfx=True, skip_data_checks=True)
            elif model_type == ModelType.SPLINE_MIXED_EFFECTS:
                if label == 'unseen_locations':
                    y_pred = scam_lib.predict_scam(model, newdata=test_set, type="response", exclude="s(ihme_loc_id)")
                else:
                    # Ensure test set has same categories as training set
                    test_set['ihme_loc_id'] = pd.Categorical(test_set['ihme_loc_id'], categories=train_levels)
                    y_pred = scam_lib.predict_scam(model, newdata=test_set, type="response")
                # unused categories generate nan groupings, convert to string
                test_set['ihme_loc_id'] = test_set['ihme_loc_id'].astype(str)
                    

            #y_pred = model.predict(test_set, verify_predictions=False, use_rfx=True, skip_data_checks=True)
            temp_test = test_set.copy()
            temp_test['predicted_probs'] = y_pred
            assert not temp_test['predicted_probs'].isnull().any(), "Predicted probabilities contain NaN values."
            test_sets.append(temp_test)
        fold_test_set = pd.concat(test_sets, ignore_index=True)
        # Point-wise metrics
        m = calculate_metrics(fold_test_set[target_measure], fold_test_set['predicted_probs'])
        
        # Aggregate (Prevalence) metrics
        # We group by location-year to see how well we predict the "rate"
        group_stats = fold_test_set.groupby([location_variable, year_variable]).agg({
            target_measure: 'mean',
            'predicted_probs': 'mean'
        })
        
        m['prev_mae'] = mean_absolute_error(group_stats[target_measure], group_stats['predicted_probs'])
        m['prev_rmse'] = np.sqrt(mean_squared_error(group_stats[target_measure], group_stats['predicted_probs']))
        m['fold'] = fold
        all_fold_results.append(m)

    # Aggregate results across folds
    results_df = pd.DataFrame(all_fold_results)

    summary = results_df.agg({
        'log_loss': ['mean', 'std'],
        'brier_score': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'auc_pr': ['mean', 'std'],
        'prev_mae': ['mean', 'std'],
        'prev_rmse': ['mean', 'std']
    }).unstack().to_frame().T
    
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

    #Add RMSE when compared to GBD
    merged_gbd_data = merge_gbd_data(target_measure, df).dropna()
    summary['gbd_rmse'] = np.sqrt(mean_squared_error(merged_gbd_data['gbd'], merged_gbd_data['pred']))
    return summary

def update_results_file(summary_df: pd.DataFrame, 
                        output_path: pathlib.Path, 
                        model_version: str, 
                        submodel: str):
    """Appends the aggregated CV results to a tracking CSV."""
    
    # Transform summary into a single row
    row_data = {
        'model_version': model_version,
        'submodel': submodel,
        'timestamp': pd.Timestamp.now()
    }
    
    for _, row in summary_df.iterrows():
        row_data[f'prev_mae'] = row['prev_mae_mean']
        row_data[f'auc_pr'] = row['auc_pr_mean']
        row_data[f'ece'] = row['ece_mean']
        row_data[f'log_loss'] = row['log_loss_mean']
        row_data[f'brier_score'] = row['brier_score_mean']
        row_data[f'prev_rmse'] = row['prev_rmse_mean']
        row_data[f'gbd_rmse'] = row['gbd_rmse']

    new_row = pd.DataFrame([row_data])
    
    if output_path.exists():
        all_results = pd.read_csv(output_path)
        all_results = pd.concat([all_results, new_row], ignore_index=True)
    else:
        all_results = new_row
        
    all_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def get_knot_values(df: pd.DataFrame, variable: str, spline: SplineSpecification, var_info: dict) -> np.ndarray:
    k = spline.k
    knot_strategy = spline.knot_strategy
    inner_knot_n = k - 4  # Number of inner knots
    if knot_strategy == 'quantiles':
        knot_values = np.quantile(df[variable], np.linspace(0, 1, inner_knot_n + 2)[1:-1])
        # Check for unique knot values
        if len(np.unique(knot_values)) < len(knot_values):
            # Use quantile of unique values to ensure unique knots        
            knot_values = np.quantile(df[variable].unique(), np.linspace(0, 1, inner_knot_n + 2)[1:-1])
    elif knot_strategy == 'quantile_unique':
        knot_values = np.quantile(df[variable].unique(), np.linspace(0, 1, inner_knot_n + 2)[1:-1])
    elif knot_strategy == 'equal':
        knot_values = np.linspace(df[variable].min(), df[variable].max(), inner_knot_n + 2)[1:-1]
    elif knot_strategy == 'harrell':
        harrell_knot_values = {
            3: [0.1, 0.5, 0.9],
            4: [0.05, 0.35, 0.65, 0.95],
            5: [0.05, 0.275, 0.5, 0.725, 0.95],
            6: [0.05, 0.23, 0.41, 0.59, 0.77, 0.95],
            7: [0.025, 0.183, 0.375, 0.5, 0.625, 0.817, 0.975],
        }
        if inner_knot_n not in harrell_knot_values:
            raise ValueError(f"Harrell knot specification not defined for {inner_knot_n} knots")
        knot_values = np.quantile(df[variable], harrell_knot_values[inner_knot_n])
    elif knot_strategy == 'custom_knots':
        if spline.knots is None:
            raise ValueError("Custom knot strategy requires user-provided knot values.")
        # Apply transformation
        knot_values = var_info[variable]['transformer'](np.array([spline.knots])).flatten()
    else:
        raise ValueError(f"Unknown knot specification: {spline.knot_strategy}")
    data_min = df[variable].min()
    data_max = df[variable].max()
    core_knots = [data_min, *knot_values, data_max]

    spacing = np.mean(np.diff(core_knots))
    
    full_knots = [
        data_min - (spacing * 3), data_min - (spacing * 2), data_min - spacing,
        *core_knots,
        data_max + spacing, data_max + (spacing * 2), data_max + (spacing * 3)
    ]
    return full_knots