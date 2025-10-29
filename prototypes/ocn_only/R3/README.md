## Ocn-only R3
This is 6h, 1IC, coarseTop configuration. 
This 6h emulator has more uniformly-spaced vertical levels -- to
avoid the multicollinarity issue.  Additionally in this run we use  one time
step as input unlike the 2 time steps used previously. This would presumably 
reduce the condition number of the feature matrix further. 

## Configuration
The hyperparameter values are,

peak learning rate: 1e-3 
weight decay      : 0.1 
hidden layers     : 1 
latent size       : 512 
gnn message steps : 16 
batch size        : 16 
num epochs	  : 64 
use half precision: False 
mesh size         : 5

## Outcomes
[T1M] First training attempt. 
The result was overfitting, so I killed the job after epoch 50 without running 
the inference, which would use the last overfitted checkpoint. I'm not computing 
the inference as of now, but if needed, we can use a checkpoint before overfitting, 
say, epoch 10, to produce inferences.

I also used latent size = 192 in this configuration and ran a long training, but
the training started spitting nan values after epoch 16. Below are the take
aways from this run:
- Training loss suddenly jumps to NaN exactly at epoch 16.
- Validation loss flattens just before that, showing mild overfitting pressure
  but not catastrophic.
- Learning rate is still decaying smoothly (cosine curve), so it's not an LR
  spike issue.  This confirms that the root cause is not overfitting, but rather
  numerical instability — likely due to activation explosions, gradient spikes, or
  poor interactions with the optimizer.

Possible solutions:
- Set clip_by_global_norm(1.0) instead of 32.0
- Lower peak LR slightly -- maybe try 3e-4 instead of 1e-3 for ocean-only
  training
- Use drop out or add small noise to the inputs to generalize better
