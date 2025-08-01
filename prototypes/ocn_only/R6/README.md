# Ocn-only run R6
In this run, we adopt the most successful configuration of 24h ocn-only model,
i.e., R5, and test it for 2 ICs, i.e, the ICs covering 48 hours as opposed to
24h in R5. The hyperparameter values are as follows:
- input time steps = 2
- more uniformly-spaced vertical levels 
- neurons per hidden layer = 256
- gnn_msg_passing_steps = 12
- peak learning rate = 1e-3
- num hidden layers = 1
- weight decay = 0.1

# Configurations:
[T1M] Full training with everything the same as R5 except the number of ICs.
