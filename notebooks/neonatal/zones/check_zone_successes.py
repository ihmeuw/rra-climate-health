import os
import re
import pandas as pd

converge_df = pd.DataFrame(columns=["zone", "converged"])

SUMMARY_DIRS = [
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/zones/mo_1/model_summaries/",
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/zones/mo_3/model_summaries/",
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/zones/mo_6/model_summaries/",
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/zones/mo_9/model_summaries/",
]

for MO_SUMMARY_DIR in SUMMARY_DIRS:

    summaries = [f for f in os.listdir(MO_SUMMARY_DIR) if f.endswith(".txt")]
    for f in summaries:

        zone = re.findall(r"(?<=nm_).*(?=_summary)", f)[0]
        with open(MO_SUMMARY_DIR + f, "r") as infile:
            s = infile.read()

            converged = not bool(re.findall(r"Model failed to converge", s))
            converge_df = pd.concat(
                [
                    converge_df,
                    pd.DataFrame.from_records([{"zone": zone, "converged": converged}]),
                ],
                ignore_index=True,
            )
converge_df["converged"].value_counts()

converge_df[converge_df["converged"] == True]
