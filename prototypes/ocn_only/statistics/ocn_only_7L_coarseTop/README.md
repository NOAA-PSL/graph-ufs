# Description
The statistics computed here belongs to the ocn-only configuration with 
reorganized vertical levels. The 3D ocean variables have vertical interfaces at 
z_i = (0, 
        1,
        21,
        75,
        120,
        200,
        350,
        500,)
The above eight interfaces lead to seven layers. 

Note that any statistics computed prior to this for ocn, land, and sea ice are
wrong because no masking was applied while computing them. This would damp mean
for positive-valued variables like salinity and bump their stddev artificially.
Other variables with mixed positive/negative values would not be affected much,
but it is still nice to have everything correct and consistent. 

Although each of 6h/ 24h/ and 48h/ folders have ocn.* and atm.* folders, the 
atm.* folders only contain the statistics for atm variables used in the ocn-
only configuration, not for all atm variables. The same also goes for 3D atm
variables which only use top-two near-surface levels as required in ocn-only 
config.   
