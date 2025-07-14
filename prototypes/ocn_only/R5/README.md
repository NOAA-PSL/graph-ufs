# Ocn-only run R5
In this run, we adopt the most successful configuration of 6h ocn-only model and
test it in 24h configuration. In the hyperparameter testing done in R4, the best
ocn-only configuration was found for:
- input time steps = 1
- more uniformly-spaced vertical levels 
- neurons per hidden layer = 256
- gnn_msg_passing_steps = 12
- peak learning rate = 1e-4
- num hidden layers = 1
- weight decay = 0.1

We test a 24 hour ocean-only model using the above configuration. Here the
ocn-model would have only one time step as input, i.e., 24 hour prior state, and
would predict tendencies for the 24-hour forward state.

# Configurations:
[T0x] All hyperparameters the same as the most optimized one. The training was
      smooth up to the 3 epochs and there was a small gap between the training
      and validation losses. However, increasing the LR to 1e-3 showed a better
      training as tested in T1x
[T1x] This uses the most optimized hyperparameters as found in T3M except the 
      learning rate, which is kept equal to the original value, 1e-3. This is 
      because it looks like the training may benefit from a higher learning rate
      here due to increased signal to noise ratio -- because of the 24h
      stepping.
[T1M] A full-blown training of T1x.
       
