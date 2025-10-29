# Ocn-only R1
This is 24h, 2IC, fineTop configuration. This run was originally done after
numerous failed attempts to emulate 6h increments in the ocean. The hypothesis 
was that there is not enough signal to learn in such a short time step and 
therefore the model is overfitting. Presumabely, 24hr increments would have 
more coherent signals to learn and emulate. I have however adapted the original
run to the new design to make a coherent structure of the entire ocn-only
emulation.
 
## Base Configuration
The base hyperparameters of this configuration are as follows:
latent size   : 256
peak LR       : 1e-3
hidden layers : 1
weight_decay  : 0.1
gnn_msg_steps : 12
mesh_size     : 5
batch_size    : 16

Note that the peak LR here is 10 times higher than the R0 emulator. 

## Prototypes and Outcomes
[T1M] The first training attempt with the base setting. 
