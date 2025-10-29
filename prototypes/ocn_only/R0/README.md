# Ocn-only R0
This is 6h, 2IC, fineTop configuration, the first attempt to build an ocn-only
emulator. The initial run was different but I moulded it to the new setup
afterwards, re-generated the dataset, and re-trained everything. 

## Base Configuration
The base hyperparameters of this configuration are as follows:
latent size   : 256
peak LR       : 1e-4
hidden layers : 1
weight decay  : 0.1
gnn_msg_steps : 12
mesh size     : 5
batch size    : 16

## Prototypes and Outcomes
[T1M] The first training attempt with the base setting.
