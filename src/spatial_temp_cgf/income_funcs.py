import pandas as pd
import paths

def load_binned_income_distribution_proportions(fhs_location_id = None, model = None, measure = None, year_id = None):
    if model is not None:
        model_identifier = model.model_identifier
        ldipc_distribution_bin_proportions_filepath = paths.MODEL_ROOTS / model_identifier / f'ldipc_bin_proportions.parquet'
    elif measure is not None:
        filepath = paths.LDIPC_DISTRIBUTION_BIN_PROPORTIONS_DEFAULT_FILEPATH_FORMAT.format(measure = measure)
        load_filters = []
        if fhs_location_id: load_filters.append(('fhs_location_id', '==', fhs_location_id))
        if year_id : load_filters.append(('year_id', '==', year_id))
        return pd.read_parquet(filepath, filters = load_filters)
    else:
        raise ValueError("Either model or measure must be specified")

