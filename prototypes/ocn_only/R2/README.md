# Ocn-only R2
This is 6h, 2IC, coarseTop configuration.
In this run, we have removed too many layers near the surface ocean as it was
possibly leading to ill-conditioned feature matrix and therefore the grid
artifacts we were seeing in the inferences. In the new setup, the vertical
levels are more uniformly spaced than before  and do not possess as much
vertical correlation as before. Note, however that this prototype still uses
two time steps as IC, not one. So the training and validation dataset on
scratch has two ICs as inputs. 

A brief literature review on the vertical
levels of ocean-only emulators suggested that the existing emulators do not
possess too much vertical resolution near the surface ocean -- as done in
physical ocean modeling. For e.g., GLONET possess only 3 layers in the top 100m
ocean. Although OLA probably had more vertical resolution around the surface but
full credibility of this model is yet to be established.  
 
## Configurations and Outcomes
I deleted all previous runs and outcomes of this prototype before running a
fresh one with the gaussian noise. Naturally it makes sense to run a vesion of
this prototype without any noise and then run one with varying levels of noise.

[T1M] The hyperparameters of this configuration are based on the hyperparameter
testing done in R4 prototype.
latent size   : 256
peak LR       : 1e-4
hidden layers : 1
weight decay  : 0.1
gnn_msg_steps : 12
mesh size     : 5
batch size    : 16
This is a dry run with zero gaussian noise amplitude.

[T2M] This configuration uses the same hyperparameter combination as above but
uses gaussian noise with 1% of the std as its amplitude. Note that the std used
for computing the noise amplitude is constant across the space.  

[T3M] Same as above but with 0.1% gaussian noise amplitude (noise amplitude
constant in space).

[T4M] Same as above but with 5% gaussian noise (constant in space amplitude).

[T5M] In the next series of runs, spatial std would be used for computing the
noise amplitude. This has shown more promise in the 1IC, 6hr time step model
(R4) and must be tested here for 2 ICs, which is more suited for this
experiment. Here the noise amplitude is 1% of the std.
     The training showed an increase in the training and the validation loss
compared to the no noise case. However, this doesn't necessarily mean a degraded
inference. 

[T6M] Same as T5M except that the noise amplitude is 0.5% of the std of the
channels, i.e., noise_std_as_fraction = 0.005  

[T7M] Same as above but using 5% of the standard deviation of the channels,
i.e., noise_std_as_fraction = 0.05

[T8M] Same as T1M, i.e., dry run without any gaussian noise addition, but with
mesh_size=6, i.e., 6 levels of mesh refinement just like in GraphCast.
