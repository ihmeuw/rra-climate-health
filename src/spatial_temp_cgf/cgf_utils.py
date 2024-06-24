import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def bin_column(df, col, nbins, label_strategy = None, bin_strategy = 'quantiles', retbins = False):
    precision = 3 if nbins > 10 else 1
    drop_dupes = nbins > 10
    if label_strategy == 'means':
        label_strategy = [f'{col}_bin_{i}' for i in range(nbins)] #not quite right
    if bin_strategy == 'quantiles':
        binned, bins = pd.qcut(df[col], q=nbins, labels=label_strategy, duplicates='drop', precision = precision, retbins=True)
    elif bin_strategy == 'equal':
        binned, bins = pd.cut(df[col], bins=nbins, labels=label_strategy, include_lowest = True, retbins=True)
    elif bin_strategy == 'readable_5':
        bins = [df[col].min(), 20] + list(range(22, 2+int(np.ceil(df[col].max())), 2))#list(range(5*int(df[col].min() // 5), 5+5*int(df[col].max() // 5), 5))
        binned = pd.cut(df[col], bins = bins, labels = label_strategy, include_lowest = True, right=False)
    elif bin_strategy == '0_1_more':
        _ , otherbins = pd.qcut(df.loc[df[col] > 1, col], q=nbins-2, retbins = True, duplicates = 'drop', precision = precision)
        otherbins[-1] = otherbins[-1]+1
        bins = [0,1] + list(otherbins)
        binned = pd.cut(df[col], bins=bins, labels = label_strategy, include_lowest = True, right=False)
    elif bin_strategy == '0_more':
        _ , otherbins = pd.qcut(df.loc[df[col] >= 1, col], q=nbins-2, retbins = True, duplicates='drop', precision = precision)
        otherbins[-1] = otherbins[-1]+1
        bins = [0,] + list(otherbins)
        binned = pd.cut(df[col], bins=bins, labels = label_strategy, include_lowest = True, right=False)
    elif bin_strategy == '0_more_readable':
        bin_interval = 10
        bins = [0,1] + list(range(bin_interval, int((np.ceil((df[col].max()/bin_interval))+1)*bin_interval), bin_interval))
        binned = pd.cut(df[col], bins = bins, labels = label_strategy, include_lowest = True, right=False)
    elif bin_strategy == 'custom_daysover':
        bins = [0,1,5,10,20,30,40,50,60,70,80,90,100,110,120,250]
        binned = pd.cut(df[col], bins = bins, labels = label_strategy, include_lowest = True, right=False)
    else:
        raise ValueError('bin_strategy must be either "quantiles" or "equal"')
    if retbins:
        return bins, binned
    return binned

def group_and_bin_column(df, group_cols, bin_col, nbins, bin_strategy = 'quantiles', 
        label_strategy = None, result_column = None, keep_count = False,
        retbins = False):
    if result_column == None:
        result_column = bin_col + '_bin'
    grouped_df = df.groupby(group_cols + [bin_col], as_index=False).size()
    if not keep_count:
        grouped_df = grouped_df.drop(columns=['size'])

    if retbins:
        bins, binned_column = bin_column(grouped_df, bin_col, nbins, label_strategy, bin_strategy, retbins = True) 
    else:
        binned_column = bin_column(grouped_df, bin_col, nbins, label_strategy, bin_strategy) 

    grouped_df[result_column] = binned_column
    if retbins:
        return bins, pd.merge(df, grouped_df, how='left')
    else:
        return pd.merge(df, grouped_df, how='left')
    
    
def group_and_bin_column_definition(df, bin_col, bin_category, nbins, bin_strategy = None, result_column = None, retbins = False):
    if bin_category == 'household':
        group_cols = ['nid', 'hh_id', 'psu', 'year_start']
        bin_strategy = 'quantiles' if bin_strategy is None else bin_strategy
    if bin_category == 'location':
        group_cols = ['lat', 'long']
        bin_strategy = 'quantiles' if bin_strategy is None else bin_strategy
    if bin_category == 'country':
        group_cols = ['iso3']
        bin_strategy = 'quantiles' if bin_strategy is None else bin_strategy
    return group_and_bin_column(df, group_cols, bin_col, nbins, 
            bin_strategy = bin_strategy, result_column = result_column, retbins = retbins)

printable_names = {
        'income_per_day_bin':'Daily Income (2010 USD)',
        'temp_bin': 'Yearly Temperature (Mean)',
        'temp_bin_quants' : 'Mean Yearly Temperature',
        'temp_avg_bin' : 'Mean Temperature (5 year)',
        'precip_bin': 'Yearly Precipitation',
        'precip_avg_bin': 'Mean Yearly Precipitation (5 year avg)',
        'over30_bin': 'Days over 30° C',
        'over30_avg_bin': 'Days over 30° C (5 year avg)',
        'stunting': 'Stunting',
        'wasting':'Wasting',
        'underweight':'Underweight',
        'over30_avgperyear_bin' : 'Lifetime Yearly Average Days Above 30°C',
        'over30_birth_bin' : 'Days over 30 in year of birth', 
        'temp_diff_birth_bin': 'temp diff birth',
        'bmi': 'BMI',
        'low_adult_bmi' : 'Low adult BMI',
        'cgf_value': 'Stunting',
        'adjusted_pred' : 'Controlling for Country Effects',
        'predict_nocountry': 'Prediction without country',
        'residual': 'residual',
        'grid_coef' : 'dummy coefficient',
        'gdppc': 'GDP per capita',
        'ldi_pc_pd': 'Lag-distributed income per day',
        'ldi_pc_pd_bin': 'Lag-distributed income per day',
        'ldipc': 'Lag-distributed income per day',
    }


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

def plot_heatmap(df, temp_col, wealth_col = 'income_per_day_bin', country = None, year = None, 
    margins = False, filter = None, value_col = 'cgf_value', title_addin = '', file=None, 
    pdf_handle = None, vmin = None, vmax = None, counts = True, figsize = (12, 8)):  
    plot_df = df.copy()
    temp_value_col = temp_col.replace("_bin", "")
    temp_value_col = temp_value_col if temp_value_col in plot_df.columns else 'temp'

    wealth_value_col = wealth_col.replace("_bin", "")

    x_ticks = range(plot_df[temp_col].nunique() + 1)
    x_labs =  plot_df.groupby(temp_col)[temp_value_col].min().apply(lambda x: f"{x:.1f}").values.tolist() + \
        [f"{df[temp_value_col].max():.1f}"]
    x_labs = [x for x in x_labs if x != 'nan']

    y_ticks = range(plot_df[wealth_col].nunique() + 1)
    y_labs =  plot_df.groupby(wealth_col)[wealth_value_col].min().apply(lambda x: f"{x:.1f}").values.tolist() + \
        [f"{plot_df[wealth_value_col].max():.1f}"]
    y_labs = [x for x in y_labs if x != 'nan']

    if filter is not None:
        plot_df = plot_df.query(filter)
    if country:
        plot_df = plot_df[plot_df['iso3'] == country]
    if year:
        plot_df = plot_df[plot_df['year_start'] == year]

    #plot_df = plot_df.rename(columns = printable_names)

    #value_col = printable_names[value_col] if value_col in printable_names.keys() else value_col

    pivot_table_mean = plot_df.pivot_table(values=value_col, index=wealth_col, 
        columns=temp_col, aggfunc='mean', dropna=False, margins=margins, observed=False)
    pivot_table_count = plot_df.pivot_table(values=value_col, index=wealth_col, 
        columns=temp_col, aggfunc='count', dropna=False, margins=margins, observed=False)

    fig, ax = plt.subplots(figsize=figsize)

    if vmax:
        colorbin_interval = 0.025
        boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
        cmap = plt.get_cmap('RdYlBu_r', len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        sns.heatmap(pivot_table_mean, annot=True, fmt=".2f", annot_kws={"size": 14, "weight":'regular'}, #cmap='RdYlBu_r',
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={"ticks": boundaries})
    else:
        sns.heatmap(pivot_table_mean, annot=True, fmt=".2f", cmap='RdYlBu_r')

    # Overlay the counts on the heatmap
    if counts:
        for i, row in enumerate(pivot_table_mean.values):
            for j, value in enumerate(row):
                plt.text(j+0.5, i+0.6, f'\n({pivot_table_count.values[i][j]})', 
                        ha="center", va="center");

    label_naming = lambda x: printable_names[x] if x in printable_names.keys() else x

    ax.set_xlabel(label_naming(temp_col), fontsize=16)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labs, rotation = 45)

    ax.set_ylabel(label_naming(wealth_col), fontsize=16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labs)

    cbar = ax.collections[0].colorbar
    cbar.set_label(label_naming(value_col), size=16)
    ax.set_title(title_addin, fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # plt.title((f'{printable_names[wealth_col]} x {printable_names[temp_col]} \n x {title_addin}{value_col}' 
    # f'(Mean Proportion & Count) {country if country else ", All Locations"}'));
    if file:
        plt.savefig(file, dpi=300, bbox_inches="tight")
    if pdf_handle:
        pdf_handle.savefig(fig)
    else:
        plt.show();
    plt.close()


def get_coeff_df(model):
    coeff_df= pd.DataFrame({'variable': model.params.index.str.extract(r'C\(([^\)\,]+)').values.flatten(),
        'value' : model.params.index.str.extract(r'\[T\.(.+)\]$').values.flatten(),
        'coef': model.params.values, })
    return coeff_df


def get_adjusted_model_predictions(model, df):
    coeff_df = get_coeff_df(model)
    for var in coeff_df.variable.dropna().unique():
        missing_bin = set(df[var].unique()) - set(coeff_df[coeff_df.variable == var].value.unique())
        assert(len(missing_bin) == 1)
        missing_bin = missing_bin.pop()
        #using concat instead of append
        coeff_df = pd.concat([coeff_df, pd.DataFrame({'variable': var, 'value': missing_bin, 'coef': 0}, index=[0])])

    coeff_df = coeff_df.reset_index(drop=True)
    coeff_df.loc[coeff_df.variable.isna(), 'variable'] = 'intercept'

    for var in coeff_df.variable.dropna().unique():
        var_df = coeff_df[coeff_df.variable == var].copy()
        if var_df.value.isna().all():
            assert(len(var_df) == 1)
            df[f'{var}_coef'] = var_df.coef.values[0]
            continue
        
        var_df[f'{var}_mean'] = var_df.coef.mean()
        var_df = var_df.rename(columns = {'coef':f'{var}_coef', 'value': var})
        var_df = var_df.drop(columns = 'variable')
        df = df.merge(var_df.rename(columns = {'coef':f'{var}_coef'}), on=var, how='left')
    
    df['linear_combination'] = df['intercept_coef'] + df['grid_cell_coef'] + df['iso3_coef'] - df['iso3_mean']
    df['predict_nocountry'] = 1 / (1 + np.exp(-df['linear_combination']))
    df['predict_level'] = 1 / (1 + np.exp(-(df['intercept_coef'] + df['grid_cell_coef'])))
    df['predict'] = (1 / (1 + np.exp(-(model.fittedvalues))))
    df['adjusted_residuals'] = df['cgf_value'] - df['predict_nocountry']
    df['residual'] = model.resid_response
    df['adjusted_pred'] = df['predict_nocountry'] + df['residual']
    df['test'] = (1 / (1 + np.exp(-df['intercept_coef'] - df['grid_cell_coef'] -df['iso3_coef'] )))
    return df