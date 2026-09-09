import pandas as pd
from db_queries import get_location_metadata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# Set paths ####################################################################
GBD_FILE = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/gbd_prevalence/gbd_mean_child_mortality_prevalence.parquet"
INFERENCE_FILE_09_02 = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_09_02.01/ssp245.parquet"  # NIDS dropped entirely
INFERENCE_FILE_09_04 = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_09_04.01/ssp245.parquet"  # wealth index fixed as floats
INFERENCE_FILE_09_08 = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_09_08.02/ssp245.parquet"  # No birth_year variable
INFERENCE_FILE_08_28 = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_08_28.02/ssp245.parquet"
PLOT_OUTPUT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/"

# Load and format ##############################################################


def get_mean_draw(data):
    draw_cols = [col for col in data.columns if col.startswith("draw_")]
    mean_draw = data[draw_cols].mean(axis=1)
    data_out = data[[col for col in data.columns if "draw_" not in col]]
    data_out["prevalence"] = mean_draw
    return data_out


gbd_df = (
    pd.read_parquet(GBD_FILE)
    .reset_index()
    .rename(columns={"gbd_mean_prevalence": "prevalence"})
)

df_09_02 = (
    pd.read_parquet(INFERENCE_FILE_09_02)
    .reset_index()
    .drop(columns=["age_group_id", "scenario"])
)
df_09_04 = (
    pd.read_parquet(INFERENCE_FILE_09_04)
    .reset_index()
    .drop(columns=["age_group_id", "scenario"])
)
df_09_08 = (
    pd.read_parquet(INFERENCE_FILE_09_08)
    .reset_index()
    .drop(columns=["age_group_id", "scenario"])
)
df_08_28 = (
    pd.read_parquet(INFERENCE_FILE_08_28)
    .reset_index()
    .drop(columns=["age_group_id", "scenario"])
)


df_09_02 = get_mean_draw(df_09_02)
df_09_04 = get_mean_draw(df_09_04)
df_09_08 = get_mean_draw(df_09_08)
df_08_28 = get_mean_draw(df_08_28)

intersection_years = list(set(df_09_02["year_id"]).intersection(set(gbd_df["year_id"])))
df_08_28 = df_08_28[df_08_28["year_id"].isin(intersection_years)].reset_index(drop=True)
df_09_02 = df_09_02[df_09_02["year_id"].isin(intersection_years)].reset_index(drop=True)
df_09_04 = df_09_04[df_09_04["year_id"].isin(intersection_years)].reset_index(drop=True)
df_09_08 = df_09_08[df_09_08["year_id"].isin(intersection_years)].reset_index(drop=True)

gbd_df = gbd_df[gbd_df["year_id"].isin(intersection_years)].reset_index(drop=True)

gbd_df.rename(columns={"prevalence": "prevalence_gbd"}, inplace=True)
df_08_28.rename(columns={"prevalence": "prevalence_initial"}, inplace=True)
df_09_02.rename(columns={"prevalence": "prevalence_NIDs_dropped"}, inplace=True)
df_09_04.rename(columns={"prevalence": "prevalence_index_fixed"}, inplace=True)
df_09_08.rename(columns={"prevalence": "prevalence_no_birth_year"}, inplace=True)

merged_df = gbd_df.merge(df_08_28, on=["location_id", "year_id", "sex_id"], how="left")
merged_df = merged_df.merge(
    df_09_02, on=["location_id", "year_id", "sex_id"], how="left"
)
merged_df = merged_df.merge(
    df_09_04, on=["location_id", "year_id", "sex_id"], how="left"
)
merged_df = merged_df.merge(
    df_09_08, on=["location_id", "year_id", "sex_id"], how="left"
)

# Add location names
locs = get_location_metadata(location_set_id=39, release_id=16)

merged_df = merged_df.merge(
    locs[["location_id", "location_name"]], on="location_id", how="left"
)

# Plot time series for each location and save to pdf ###########################

color_dict = {
    "prevalence_gbd": "blue",
    "prevalence_initial": "red",
    "prevalence_NIDs_dropped": "green",
    "prevalence_index_fixed": "purple",
    "prevalence_no_birth_year": "orange",
}
sex_id_dict = {1: "Male", 2: "Female"}

merged_df["Sex"] = merged_df["sex_id"].map(sex_id_dict)

plot_df = merged_df[merged_df["prevalence_NIDs_dropped"].notnull()].copy()

plot_locs = (
    plot_df[["location_id", "location_name"]]
    .drop_duplicates("location_id")
    .to_dict("records")
)

plot_path = f"{PLOT_OUTPUT}time_series_by_location_all_series.pdf"
sex_order = [(1, "Male"), (2, "Female")]
locations_per_page = 6

with PdfPages(plot_path) as pdf:
    page_starts = range(0, len(plot_locs), locations_per_page)
    for page_start in tqdm(page_starts, desc="Creating plot pages"):
        page_locs = plot_locs[page_start : page_start + locations_per_page]
        fig, axes = plt.subplots(
            nrows=locations_per_page,
            ncols=2,
            figsize=(14, 24),
            squeeze=False,
        )

        for row, location in enumerate(page_locs):
            location_id = location["location_id"]
            location_name = location["location_name"]
            for col_idx, (sex_id, sex_name) in enumerate(sex_order):
                ax = axes[row, col_idx]
                loc_sex_df = plot_df[
                    (plot_df["location_id"] == location_id)
                    & (plot_df["sex_id"] == sex_id)
                ]
                for prevalence_col, color in color_dict.items():
                    if prevalence_col in loc_sex_df.columns:
                        ax.plot(
                            loc_sex_df["year_id"],
                            loc_sex_df[prevalence_col],
                            label=prevalence_col,
                            color=color,
                        )
                ax.set_title(f"{location_name} - {sex_name}")
                ax.set_xlabel("Year")
                ax.set_ylabel("Prevalence")
                ax.legend()

        for row in range(len(page_locs), locations_per_page):
            for col_idx in range(2):
                axes[row, col_idx].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
