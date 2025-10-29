# Ocn-only R11
This is 6h, 1IC, fineTop configuration.
In this run, we train a 6 hour ocean-only emulator using the same hyperparam 
configuration as R4 but with the fineTop setting.
Note that, here the training would be using MSE only, not using the Mahalanobis
loss. 

# Configurations:
[T1M] The intended first full-blown training. This is not done yet because I
created this prototype directory to compute the statistics for 6h, fineTop
config using the revised code, i.e., the code using a mask over the oceans
before computing the statistics. The training of this emulator is not necessary
at this point for the paper.   
       
