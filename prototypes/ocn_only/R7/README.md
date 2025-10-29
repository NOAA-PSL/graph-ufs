# Ocn-only R7
This is 24h, 1IC, coarseTop configuration but with Bathymetry.
In this run, we adopt the most successful configuration of 24h ocn-only model and
test it with bathymetry. The hypothesis is that perhaps bathymetry would lead to
better training as it may allow a better handling of the coastal regions. All
the hyperparameters are kept the same.

# Configurations:
[T1M] A full-blown training of this prototype.
[T2M] In the above run, it was found that the training loss was still going
down. So this run uses 80 epochs instead of 40; rest everything is the same. 
