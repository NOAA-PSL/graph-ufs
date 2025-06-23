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
would predict tendecines for the 24-hour later state.      
