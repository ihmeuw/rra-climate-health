"""
R cannot save data frames to parquet with row names as index.
"""

import pandas as pd

# Random effects
INFILE = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/inference_format/nnm_1_mo_q95_ranef.csv"
infile = pd.read_csv(INFILE, index_col=0)

OUTFILE = INFILE.replace(".csv", ".parquet")
print(OUTFILE)
infile.to_parquet(OUTFILE, index=True)

# Coefficients
INFILE = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/inference_format/nnm_1_mo_q95_coefs.csv"
infile = pd.read_csv(INFILE, index_col=0)

OUTFILE = INFILE.replace(".csv", ".parquet")
print(OUTFILE)
infile.to_parquet(OUTFILE, index=True)
