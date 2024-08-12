import pandas as pd
from pathlib import Path
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
from io import BytesIO

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

def plot_forecast_prevalence(merged, model_spec, grouping_col=None):
    grouping_col = [grouping_col] if grouping_col else []
    plot_df = merged.groupby(['year_id', 'scenario'] + grouping_col).agg({'affected': 'sum', 'population': 'sum', 'delta':'sum'}).assign(prev=lambda x: x.affected / x.population)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='year_id', y='prev', hue='scenario', marker='o', ax=ax)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Prevalence")
    
    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.figtext(0.5, -0.05, model_spec, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.close()
    return fig

def create_table(data, title, as_integers = False):
    styles = getSampleStyleSheet()

    # Generate the header rows
    multi_index = data.columns
    upper_header = []
    lower_header = []

    for col in multi_index:
        if isinstance(col, tuple):
            upper_header.append(col[0])
            lower_header.append(col[1])
        else:
            upper_header.append(col)
            lower_header.append('')

    # Remove duplicate entries for merged cells in the upper header
    prev_col = None
    for i, col in enumerate(upper_header):
        if col == prev_col:
            upper_header[i] = ''
        else:
            prev_col = col

    # Combine header rows and data
    if as_integers:
        data_list = [upper_header, lower_header] + data.applymap(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x).values.tolist()
    else:
        data_list = [upper_header, lower_header] + data.values.tolist()

    # Create the table
    table = Table(data_list)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 2), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])

    # Add spanning for multi-level headers
    for i, col in enumerate(upper_header):
        if col:
            start = i
            end = start
            while end < len(upper_header) - 1 and upper_header[end + 1] == '':
                end += 1
            if end > start:
                style.add('SPAN', (start, 0), (end, 0))

    table.setStyle(style)
    return [Paragraph(title, styles['Heading2']), table, Spacer(1, 20)]

def get_cumulative_differences(merged, target_years=[2050, 2100], grouping_col=None):
    grouping_col = [grouping_col] if grouping_col else []
    merged['ref_affected'] = merged['ref_prev'] * merged['population']
    dfs = []
    
    for target_year in target_years:
        cumsum = merged.groupby(['year_id', 'scenario'] + grouping_col).agg({'affected': 'sum', 'ref_affected': 'sum', 'population': 'sum', 'delta': 'sum'})
        cumsum = cumsum.query("year_id <= @target_year").groupby(['scenario'] + grouping_col).sum()
        cumsum['target_year'] = target_year
        if grouping_col:
            cumsum['geography_level'] = grouping_col[0]
            cumsum = cumsum.reset_index().rename(columns={grouping_col[0]: 'geography'})
        else:
            cumsum['geography_level'] = 'Global'
            cumsum['geography'] = 'Global'
            cumsum = cumsum.reset_index()
        dfs.append(cumsum)
    result_df = pd.concat(dfs)

    show_df = result_df.pivot_table(index=['geography_level', 'geography'], columns=['scenario', 'target_year'], values='delta').reset_index().drop(columns=['ssp245', 'ssp585'])
    show_df['geography_level'] = show_df['geography_level'].replace({'Global': 'Global', 'super_region_name': 'Super Region', 'region_name': 'Region'})
    show_df = show_df.rename(columns={'ssp119': 'SSP1-19', 'constant_climate': 'Constant Climate', 'geography_level': 'Geography Level', 'geography': 'Geography'}).round()
    return show_df

def save_plot_to_bytes(fig=None):
    buf = BytesIO()
    if fig is None:
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    else:
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def create_inference_diagnostics_report(output_path, measure, results_version, model_version):
    cm_data = ClimateMalnutritionData(output_path / measure)

    output_path = cm_data.results / results_version / f"forecast_diag.pdf"

    doc = SimpleDocTemplate(output_path.as_posix(), pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Inference Diagnostics", styles['Heading1']))
    elements.append(Spacer(1, 30))

    forecast = cm_data.load_forecast(results_version)

    cumulative_differences = get_cumulative_differences(forecast)
    elements.extend(create_table(cumulative_differences, "Cumulative Differences", as_integers=True))
    elements.append(Image(save_plot_to_bytes(plot_forecast_prevalence(forecast, "Prevalence")), width=600, height=450))
    elements.append(Image(save_plot_to_bytes(plot_gbd_comparison(measure, results_version, model_version, results_version)), width=600, height=450))

    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Model Diagnostics", styles['Heading2']))
    elements.append(Image(save_plot_to_bytes(plot_model_heatmaps(model_version, measure, measure)), width=600, height=200))
    elements.extend(create_table(get_coefficients_table(cm_data, model_version).reset_index(), "Coefficients", as_integers=False))
    elements.extend(create_table(get_re_table(cm_data, model_version).reset_index(), "Random Effects", as_integers=False))

    doc.build(elements)

def plot_gbd_comparison(measure, version_label, model_version, results_version, scenarios = ['ssp119'], 
    age_group_ids = [4,5], sex_ids = [1,2], year_ids = list(range(2020, 2023))):
    from spatial_temp_cgf.data_prep.location_mapping import FHS_HIERARCHY_PATH
    import pickle
    ROOT = Path(DEFAULT_ROOT)
    fhs_loc_meta = pd.read_parquet(FHS_HIERARCHY_PATH).sort_values('sort_order').reset_index(drop=True)
    def load_model(root: Path, measure: str, version: str, age_group_id: int, sex_id: int):
        with open(root / measure / 'models'/ version / f'{age_group_id}_{sex_id}.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    def load_random_effects(root: Path, measure: str, version: str,
                            version_label: str, age_group_id: int, sex_id: int, fhs_loc_meta: pd.DataFrame):
        data = load_model(root, measure, version, age_group_id, sex_id).ranef
        data['version_label'] = version_label
        data['age_group_id'] = age_group_id
        data['sex_id'] = sex_id
        data['measure'] = measure
        fhs_loc_meta['ihme_loc_id'] = fhs_loc_meta['ihme_loc_id'].str[:3]
        data = data.join(fhs_loc_meta.set_index('ihme_loc_id').loc[:, 'location_id'])

        return data.set_index(['version_label', 'measure', 'location_id', 'age_group_id', 'sex_id']).loc[:, 'X.Intercept.']


    def load_prediction(year_id: int, root: Path, measure: str, version: str, version_label: str, scenario: str):
        data = pd.read_parquet(root / measure / 'results' / version / f'{year_id}_{scenario}.parquet')
        data['year_id'] = year_id
        data['version_label'] = version_label
        data['measure'] = measure

        return data.set_index(['version_label', 'measure', 'location_id', 'year_id', 'age_group_id', 'sex_id']).loc[:, 'value'].sort_index()

    random_effects = []
    predictions = []
    for age_group_id in age_group_ids:
        for sex_id in sex_ids:
            random_effects.append(
                load_random_effects(ROOT, measure, model_version,
                                    version_label, age_group_id, sex_id, fhs_loc_meta)
            )
    for year_id in year_ids:
        for scenario in scenarios:
            predictions.append(load_prediction(year_id, ROOT, measure, results_version, version_label, scenario))
    predictions = pd.concat(predictions).sort_index().rename('pred')
    random_effects = pd.concat(random_effects).sort_index().rename('ranef')

    age_metadata = pd.read_parquet(ROOT / 'input' / 'gbd_prevalence' / 'age_metadata.parquet')
    age_metadata = age_metadata.loc[age_metadata['age_group_days_start'] >= 28]
    age_metadata = age_metadata.loc[age_metadata['age_group_years_end'] <= 5]

    population = pd.read_parquet(ROOT / 'input' / 'gbd_prevalence' / 'population.parquet')

    gbd = pd.read_parquet(ROOT / 'input' / 'gbd_prevalence' / f'{measure}.parquet')
    gbd['measure'] = measure
    gbd = gbd.groupby(['measure', 'location_id', 'year_id', 'age_group_id', 'sex_id'], as_index=False)['mean'].sum()

    gbd = gbd.merge(population)
    gbd = gbd.merge(age_metadata.loc[:, ['age_group_id', 'age_group_years_start', 'age_group_years_end']])
    gbd.loc[gbd['age_group_years_end'] <= 1, 'age_group_id'] = 4
    gbd.loc[gbd['age_group_years_start'] >= 1, 'age_group_id'] = 5
    gbd['gbd'] = gbd['mean'] * gbd['population']
    gbd = gbd.groupby(['measure', 'location_id', 'year_id', 'age_group_id', 'sex_id'])['gbd', 'population'].sum()
    gbd['gbd'] /= gbd['population']

    plot_data = predictions.to_frame().join(gbd).dropna().join(random_effects, how='left').reorder_levels(predictions.index.names)
    plot_data['has_raneff'] = plot_data['ranef'].notnull()
    plot_data['pred'] *= plot_data['population']
    plot_data['gbd'] *= plot_data['population']
    plot_data = plot_data.groupby(['version_label', 'measure', 'has_raneff', 'location_id', 'year_id'])['pred', 'gbd', 'population'].sum()
    plot_data['pred'] /= plot_data['population']
    plot_data['gbd'] /= plot_data['population']
    plot_data = plot_data.drop('population', axis=1)
    plot_data.describe()

    sns.set_style('whitegrid')

    marker = 'o'
    fig, ax = plt.subplots(figsize=(7, 5))
    
    label = f"{version_label}"
    sublabel = f""
    
    for re, color in [(True, 'mediumseagreen'), (False, 'mediumorchid')]:
        rmse = np.round(
            (
                (
                    plot_data.loc[version_label, measure, re, :, :].loc[:, 'gbd'] - plot_data.loc[version_label, measure, re, :, :].loc[:, 'pred']
                ) ** 2
            ).mean() ** 0.5,
            4
        )
        if re:
            legend_label = f'{measure}, locs in model (RMSE: {rmse})'
        else:
            legend_label = f'{measure}, locs not in model (RMSE: {rmse})'
        ax.scatter(
            plot_data.loc[version_label, measure, re, :, :].loc[:, 'gbd'],
            plot_data.loc[version_label, measure, re, :, :].loc[:, 'pred'],
            color=color,
            s=100,
            alpha=0.1,
            marker=marker,
        )
        ax.scatter(
            np.nan,
            np.nan,
            color=color,
            label=legend_label,
            s=100,
            alpha=1.,
            marker=marker,
        )
    ax.plot((0, 1), (0, 1), color='red')
    ax.set_ylabel('Prediction')
    ax.set_xlabel('GBD 2021')
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, 0.6)
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    return fig

def get_coefficients_table(cm_data, model_version):
    models = cm_data.load_model_family(model_version)
    return pd.concat([model['model'].coefs['Estimate'].rename(f"{model['age_group_id']} {model['sex_id']}") for model in models], axis=1)

def get_re_table(cm_data, model_version):
    models = cm_data.load_model_family(model_version)
    return pd.concat([model['model'].ranef.rename(columns = {'X.Intercept.': f"{model['age_group_id']} {model['sex_id']}"}) for model in models], axis=1)

def plot_model_heatmaps(model_version, measure, fig_title):
    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / measure)
    models = cm_data.load_model_family(model_version)
    df = pd.concat([model['model'].raw_data for model in models]).reset_index(drop=True)
    df['pred'] = pd.concat([pd.Series(model['model'].predict(model['model'].design_matrix, use_rfx=False, verify_predictions=False)) for model in models]).reset_index(drop=True)
    df['fits'] = pd.concat([model['model'].data for model in models]).reset_index(drop=True).fits
    df['over_30'], o30_bins = pd.cut(df.days_over_30C, [0, 1, 2, 7, 15, 30, 60, 90, 180, 367], right=False, retbins=True)
    df['ldi'], ldi_bins  = pd.qcut(df.ldi_pc_pd, 10, retbins=True)

    x_ticks = range(len(o30_bins) )
    x_labs =  [f"{x:.1f}" for x in o30_bins]

    y_ticks = range(len(ldi_bins) )
    y_labs =  [f"{x:.1f}" for x in ldi_bins]

    vmin = 0
    vmax = 0.6 if measure == 'stunting' else 0.25
    colorbin_interval = 0.05
    boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
    cmap = plt.get_cmap('RdYlBu_r', len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    fig, axes = plt.subplots(figsize=(24, 8), ncols=3)
    
    sns.heatmap(df.groupby(['ldi', 'over_30'])[measure].mean().unstack(), ax=axes[0], 
            annot=True, fmt=".2f", annot_kws={"size": 10, "weight":'regular'}, 
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={"ticks": boundaries})
    axes[0].set_title('Data')

    sns.heatmap(df.groupby(['ldi', 'over_30']).fits.mean().unstack(), ax=axes[1],
            annot=True, fmt=".2f", annot_kws={"size": 10, "weight":'regular'}, 
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={"ticks": boundaries})
    axes[1].set_title('With location random effects')

    sns.heatmap(df.groupby(['ldi', 'over_30']).pred.mean().unstack(),  ax=axes[2],
            annot=True, fmt=".2f", annot_kws={"size": 10, "weight":'regular'}, 
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={"ticks": boundaries})
    axes[2].set_title('Without location random effects')
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45,  fontsize=12)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs,  fontsize=12)
        ax.set_xlabel('Days over 30C', fontsize=12)
        ax.set_ylabel('Income', fontsize=12)
        ax.collections[0].colorbar.set_label(f'{measure.capitalize()} Prevalence', size=12)

    fig.tight_layout()
    return fig
    