"""These are paths the RRA team does not own.

They should only be accessed from the `data_prep` subpackage. Downstream pipeline code
should depend on data we manage wherever practical and load that data using the
`rra_climate_health.data` module.

"""

from pathlib import Path

#####################
# Wealth Indicators #
#####################

##########################
# Resource Tracking Data #
##########################

LDIPC_NATIONAL_FILEPATH = Path(
    "/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv"
)
LDIPC_SUBNATIONAL_FILEPATH = Path(
    "/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/admin2_ldipc_estimates.csv"
)
