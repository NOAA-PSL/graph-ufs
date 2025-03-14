# CP1 R1 run
This is the second attempt to run the CP1 configuration with full 27 years of 
REPLAY data for atmosphere, ocean, land, and sea ice variables.

## Changes relative to CP1-R0
Below changes were  made in the codebase before this run.
- All ocean, land, and sea ice variables are approapriately masked before
computing the loss function. 
- A mini stacked version of this prototype is developed to test this and 
all future code changes in order to save time and environment.
- landsea mask for 3D ocean variable is diagnosed inside the code using the 
FV regridded salinity field. Salinity was chosen because FV regridding is 
removing all nan values by default and equating them to zero. Only for 
salinity, a zero value is beyond its range and can be used to diagnose the
land area.
- land_static present within the FV3 dataset is used to mask all 2D ocean,
land and ice variables.
- Note that there are some minor differences in the top-layer ocean mask 
diagnosed through SSH and land_static -- potentially due to the interpolation
and regridding involved. But, at this point, it doesn't make sense to include 
another mask variable just for SSH.
- There are several ideas to mask sea ice variables, but all have some pros
and cons. For example, a dynamic sea ice mask would most appropriately mask
the sea ice variables in time but this would mean that we either need to make
such a dynamic sea ice mask a prognostic variable or need to diagnose it somehow.
Other ideas include using a seasonal sea ice mask or using a mask with all 
locations where sea ice has ever evolved in the past 27 years of training data.
I decided to go more theoretical and provided it the same mask as ocean 
becuase sea ice can potentially occur at any location in the ocean. 

## Outcomes
  
