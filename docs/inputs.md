# Survey Data Inputs

We use the DHS Surveys as our main source of linking a malnutrition outcome to a given zone (for climate) and household (for income).

There are three extractions that we source from:

- An extraction from the `anthropometrics` codebook, purported to have the same inputs as gbd does (from DHS)
- An extraction from the wealth team
- An extraction from the LSAE team

We link the LSAE and wealth extractions by NID, household id and year, performing cleaning where necessary, as some surveys have the strata as part of the household id, and some don't. The source of the heterogeneity in inputs given that they are coming from the same source hasn't been established.

## Income - Asset matching

In order to forecast using income, we need to transform the DHS Asset Index into a measure of income. We use Joe Dieleman's team's distributions of income.
To do that, for a given NID (that is, for a given location-year_start), we look at the unique households sampled in the survey (if there is more than one observation in the same household, we take them to experience the same income). We weight each household by the reported DHS observation weight to have an asset index distribution. We calculate to what percentile each household's asset index corresponds and we match it to the corresponding percentiles in the income distribution.
We receive the income distributions in increments of 0.1 density, however. In order to match the percentile, we interpolate using monotonic cubic splines.

However, the income distributions and asset distributions often don't have the same shape. We assume this is because the DHS surveys are unlikely to truly sample the very upper ends of the income distribution. Hence, we need fit the DHS asset distribution to a more appropriate income distribution. We do that by testing different thresholds for cutting off the upper end of the income distribution, comparing the two distributions (in a scaled CDF space) and choosing the one that minimizes the absolute sum in the difference between the distributions.

## Climate variables

We use the reported geographical coordinates in the DHS survey –even though they refer to the location of the PSU– to map the correct climate conditions for the household for all the available climate variables such as mean temperature, days above 30C, and precipitation.

We also use those coordinates to triangulate the altitude / elevation of the household.
