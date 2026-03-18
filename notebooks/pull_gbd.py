"""
Pull GBD estimates for comparison against model estimates.

Parameters:
release_id 16 is GBD 2023
"""

from db_queries import get_life_table
import os
import pandas as pd

OUTPATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/gbd_prevalence/"
df = get_life_table(release_id=16, with_ui=False, age_group_id=[42, 1])

df.to_parquet(os.path.join(OUTPATH, "gbd_life_table.parquet"))

df = pd.read_parquet(os.path.join(OUTPATH, "gbd_life_table.parquet"))

"""
925 location ids
1950-2024
3 sex IDs
mean is between 0.0001 and 88. What are units?
life_table_parameter_id betwen 1 and 8:
1	mx	mortality rate
2	ax	mean years lived in age interval among those who die in the age interval
3	qx	probability of death
4	lx	survivorship curve
5	ex	life expectancy
6	pred_ex	interpolated life expectancy from the theoretical minimum risk life table which uses the lowest observed death rate for each 5 year age group in all populations greater than 5 million individuals across all years, matched with country/age/sex/year-specific observations using rounded ax values
7	nLx	person years lived between age x and x+n
8	Tx	person-years lived in and above age interval

From hub: https://hub.ihme.washington.edu/spaces/~ermadd/pages/496927980/IHME+Glossary+updated+2026?preview=%2F496927980%2F545587771%2F1A+-+Mortality+rate+%28official%29.pptx

Mortality rate: Key considerations
•Neonatal age groups:For age intervals shorter 
than 1 year, an individualcontributes less than 1 person-year to the denominator 
(e.g. 0.019 person-years for neonatal age group 0 to 6 days) but can still contribute 
1 death tothe numerator. That is why mortality rates for neonatal age groups can be
greater than 1.
•Mortality rate(mx) is not the same as probability of death (qx). 
The probability of dying in an age interval is naturally higher for wider age
groups because an individual spends more time at risk in a wider agegroup. Mortality 
rate, however, has person-time in the denominator andtherefore accounts for time 
spent at risk, so the width of the age groupdoes not impact the magnitude of a 
mortality rate.
"""
df = df[df["life_table_parameter_id"] == 1]  # MX
df["mean"].describe()  # max is 2.17
# from the HUB, this looks like deaths per 100k person-years https://hub.ihme.washington.edu/spaces/~ermadd/pages/496927980/IHME+Glossary+updated+2026?preview=%2F496927980%2F545587771%2F1A+-+Mortality+rate+%28official%29.pptx

neo = df[df["age_group_id"] == 42]
cm = df[df["age_group_id"] == 1]

neo.to_parquet(os.path.join(OUTPATH, "neonatal_mortality.parquet"))
cm.to_parquet(os.path.join(OUTPATH, "child_mortality.parquet"))
