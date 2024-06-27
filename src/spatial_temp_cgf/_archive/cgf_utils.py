import pandas as pd

def bin_cgf_cols(in_df, nbins):
    res_df = in_df.copy()
    if 'income_per_day' in in_df.columns:
        res_df = group_and_bin_column_definition(res_df, 'income_per_day', 'household', nbins)
    if 'gdppc' in in_df.columns:
        res_df = group_and_bin_column_definition(res_df, 'gdppc', 'household', nbins)
    res_df = group_and_bin_column_definition(res_df, 'over30', 'location', 10, bin_strategy = '0_more')
    res_df = group_and_bin_column_definition(res_df, 'temp', 'location', nbins, bin_strategy = 'readable_5')
    res_df = group_and_bin_column_definition(res_df, 'temp', 'location', nbins, bin_strategy = 'quantiles', result_column = 'temp_bin_quants')
    res_df = group_and_bin_column_definition(res_df, 'temp', 'location', 25, bin_strategy = 'equal', result_column = 'temp_bin_many')
    res_df = group_and_bin_column_definition(res_df, 'precip', 'location', nbins, bin_strategy = 'quantiles')
    if 'over30_avgperyear' in in_df.columns:
        res_df = group_and_bin_column_definition(res_df, 'over30_avgperyear', 'location', nbins, bin_strategy = '0_more')
    return res_df


#TODO Assert NAs and length here
# cols_to_verify = ['over30', 'over30_bin', 'temp', 'temp_bin', 'precip', 'precip_bin', 'income_per_day', 'income_per_day_bin',]
#         #'over30_avg', 'over30_avg_bin', 'temp_avg', 'temp_avg_bin', 'precip_avg', 'precip_avg_bin']
# assert(merged_binned_df[cols_to_verify].notna().all().all())

def plot_cgf_varbins(plotcol):
    plot_binned_df = merged_binned_df.groupby([f'temp_bin_many', 'cgf_measure']).agg(
        col_value = (plotcol, 'mean'), cgf_value = ('cgf_value', 'sum'), cgf_denom = ('cgf_value', 'count')).reset_index()
    plot_binned_df = plot_binned_df.rename(columns = {'col_value' : plotcol})
    plot_binned_df['cgf_proportion'] = plot_binned_df['cgf_value'] / plot_binned_df['cgf_denom']
    plot_binned_df[f"temp_s"] = plot_binned_df[plotcol].astype(str)
    fig = px.line(plot_binned_df.sort_values(plotcol), x=plotcol, y='cgf_proportion', color = 'cgf_measure', title=f'CGF (proportion) by {plotcol}',
        hover_data = ['cgf_denom'])
    fig.show()


def get_coeff_df(model):
    coeff_df= pd.DataFrame({'variable': model.params.index.str.extract(r'C\(([^\)\,]+)').values.flatten(),
        'value' : model.params.index.str.extract(r'\[T\.(.+)\]$').values.flatten(),
        'coef': model.params.values, })
    return coeff_df

def get_adjusted_model_predictions(model, df):
    coeff_df = get_coeff_df(model)
    for var in coeff_df.variable.dropna().unique():
        missing_bin = set(df[var].unique()) - set(
            coeff_df[coeff_df.variable == var].value.unique())
        assert (len(missing_bin) == 1)
        missing_bin = missing_bin.pop()
        # using concat instead of append
        coeff_df = pd.concat([coeff_df, pd.DataFrame(
            {'variable': var, 'value': missing_bin, 'coef': 0}, index=[0])])

    coeff_df = coeff_df.reset_index(drop=True)
    coeff_df.loc[coeff_df.variable.isna(), 'variable'] = 'intercept'

    for var in coeff_df.variable.dropna().unique():
        var_df = coeff_df[coeff_df.variable == var].copy()
        if var_df.value.isna().all():
            assert (len(var_df) == 1)
            df[f'{var}_coef'] = var_df.coef.values[0]
            continue

        var_df[f'{var}_mean'] = var_df.coef.mean()
        var_df = var_df.rename(columns={'coef': f'{var}_coef', 'value': var})
        var_df = var_df.drop(columns='variable')
        df = df.merge(var_df.rename(columns={'coef': f'{var}_coef'}), on=var,
                      how='left')

    df['linear_combination'] = df['intercept_coef'] + df['grid_cell_coef'] + df[
        'iso3_coef'] - df['iso3_mean']
    df['predict_nocountry'] = 1 / (1 + np.exp(-df['linear_combination']))
    df['predict_level'] = 1 / (
                1 + np.exp(-(df['intercept_coef'] + df['grid_cell_coef'])))
    df['predict'] = (1 / (1 + np.exp(-(model.fittedvalues))))
    df['adjusted_residuals'] = df['cgf_value'] - df['predict_nocountry']
    df['residual'] = model.resid_response
    df['adjusted_pred'] = df['predict_nocountry'] + df['residual']
    df['test'] = (1 / (1 + np.exp(
        -df['intercept_coef'] - df['grid_cell_coef'] - df['iso3_coef'])))
    return df
