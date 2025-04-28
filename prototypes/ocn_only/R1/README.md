# Ocn-only run
This is the second attempt to run the ocean-only configuration with full 27 years of 
REPLAY data with 10m U-V from atmosphere as forcing. In this run, the standard deviation
of all prognostic ocean variables are bumped up to 6 times their value in R0. Hopefully,
this would calm down the loss curve.

## Outcome
* Increasing the standard deviation of 3D variables didn't help even after increasing them 
up to a factor 20.This gave the suspecion that perhaps this is not the reason for those 
sharp bumps in the loss.
* By digging the dataset, I found that the samples for which those bumps are occurring 
possess a much higher increment (I checked for salinity, but this applies for other variables 
too) compared to their neighbouring samples. This means that those bumps were due to the bad
data samples and not due to dividing by small std.
* These bad samples belonged to -----   
