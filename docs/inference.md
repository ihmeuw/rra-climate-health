# Inference

While we train using individual observations under specific climate and income circumstances, we forecast prevalences and counts for the whole planet. This discrepancy means that, in order to produce predictions, we need to manually build the prediction for every individual pixel in the globe's representation. 

Inference takes as input the coefficients from the trained model, climate forecasts for a given scenario, and income forecasts. In order to estimate prevalence for a given pixel, we take the coefficients of the climate variables and apply them to the values for the climate variables for the given scenario. Since income is given in a distribution with 10 deciles, we create an estimate for each decile in the pixel and add them up. 

In order to get a region's prevalence, we need a measure of the distribution of the population. Currently, we're using a static 2020 population raster from the Global Human Settlement Layer and calculate the proportion of population living in a given pixel and hold it static throughout the estimation period. 
We aggregate the region's pixel's prevalences to produce a single region's prevalence. In order to produce counts and wider prevalences, we use a region-specific population forecast. 
