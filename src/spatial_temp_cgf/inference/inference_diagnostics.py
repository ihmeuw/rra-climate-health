import pandas as pd
from pathlib import Path
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
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

def create_inference_diagnostics_report(output_path, measure, results_version):
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
    elements.append(Image(save_plot_to_bytes(plot_forecast_prevalence(forecast, "Prevalence")), width=500, height=300))

    doc.build(elements)

