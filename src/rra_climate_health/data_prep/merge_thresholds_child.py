import click
import pandas as pd
import sys
import os


@click.command()
@click.option("--data_file", required=True, type=str)
@click.option("--climate_file", required=True, type=str)
@click.option("--outfile_path", required=True, type=str)
def main() -> None:
    """
    data_file = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/merge_chunks/df_prev_0_mo.parquet"
    climate_file = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/climate_thresholds_for_locs.parquet"
    outfile_path = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/merged_chunks/"
    """
    filename = data_file.split("/")[-1]

    data = pd.read_parquet(data_file)
    climate_vars_df = pd.read_parquet(climate_file)
    os.makedirs(outfile_path, exists_ok=True, mode=0o777)

    df_merged = data.merge(
        climate_vars_df,
        on=["lookup_year", "lookup_month", "lat", "long"],
        how="left",
    )

    # pivot wide
    df_merged["quantile_str"] = df_merged["quantile_str"] + "_" + df_merged["suffix"]
    df_merged.drop(
        columns=["lookup_year", "lookup_month", "suffix", "longitude", "latitude"],
        inplace=True,
    )

    merge_cols = ["index", "birth_year", "birth_month", "lat", "long"]
    df_merged_wide = (
        df_merged[merge_cols + ["quantile_str", "value"]]
        .pivot_table(
            index=merge_cols,
            columns="quantile_str",
            values="value",
        )
        .reset_index()
    )
    df_merged_wide.to_parquet(outfile_path + "/" + filename)
