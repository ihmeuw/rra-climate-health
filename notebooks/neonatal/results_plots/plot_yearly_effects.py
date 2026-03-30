import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

model_summary_file = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/models/2026_03_29.01/zzz_logs/2026_03_29_09_17_12/output/18304496/training_task_template_measure-neonatal_mortality.o18304496_1"
plot_loc = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_03_29.01/plots/"
linear_model_file = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/models/2026_03_29.08/base_model.pkl"
linear_model_coefs = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/models/2026_03_29.08/base_model_coefs.parquet"

coefs = pd.read_parquet(linear_model_coefs)
with open(linear_model_file, "rb") as f:
    linear_model = pickle.load(f)

os.makedirs(plot_loc, exist_ok=True)

year_df = pd.DataFrame(columns=["year", "coef", "std_err", "z_value", "p_value"])
with open(model_summary_file, "r") as f:
    lines = f.readlines()

year_lines = [line for line in lines if line.startswith("C(birth_year)")]

rows = []  # Collect rows as dictionaries
for line in year_lines:
    try:
        parts = line.split()
        year = parts[0].split("C(birth_year)")[1].strip("[]")
        coef = float(parts[1])
        std_err = float(parts[2])
        z_value = float(parts[3])
        p_value = float(parts[4])

        rows.append(
            {
                "year": int(year),
                "coef": coef,
                "std_err": std_err,
                "z_value": z_value,
                "p_value": p_value,
            }
        )
    except (IndexError, ValueError) as e:
        print(f"Skipping malformed line: {line.strip()} - Error: {e}")

# Use pd.concat to create the DataFrame in one step
year_df = pd.concat([pd.DataFrame(rows)], ignore_index=True)


# create a plot of the coefficients with error bars
plt.figure(figsize=(10, 6))
plt.scatter(
    year_df.query("year<2023")["year"],
    year_df.query("year<2023")["coef"],
    color="blue",
    zorder=3,
)
plt.errorbar(
    x=year_df.query("year<2023")["year"],
    y=year_df.query("year<2023")["coef"],
    yerr=year_df.query("year<2023")["std_err"],
    fmt="none",
    ecolor="black",
    capsize=5,
)
plt.axhline(0, color="red", linestyle="--")
plt.title("Yearly Effects on Neonatal Mortality")
plt.xlabel("Birth Year")
plt.ylabel("Coefficient (Effect Size)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_loc, "yearly_effects.png"))
plt.show()
