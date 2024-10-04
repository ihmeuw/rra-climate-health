
# Model Specification

We use a mixed methods logistic regression. The objective is to estimate the effect of climate variables, income and other covariates on a given health outcome.

The tool allows for the following:
- Choosing to use age groups and sex as either categorical variables or to do submodels by either or both of them.
- Using any of the climate variables available
- Random effects on the intercept by location - either country-level or FHS admin-2 levels
- Income
- Other variables: SDI, year
- Interaction between a threshold climate variable and income
Other rasterized variables can be added with a few code changes. Non-rasterized variables need to be rasterized first if they vary geographically (such as country-specific variables). 

Model covariates can be expressed to undergo transformations from the raw state. These can be:
- Scaling
  - Min Max
  - Standardizing 
  - Inner 95: Scale but setting the top and bottom values at the 0.25 and 0.975 percentile of the covariate values' distribution
- Binning
- Masking

In order to specify all these, refer to the example model specifications.

Model training follows the specification provided, transforms the data and feeds it to a Lmer model.