# Description
This prototype directory belongs to the ocean-only configuration, where the
prognostic variables only belong to the oceans. Several atmospheric variables
are used as well but only as forcing. The base configuration of this prototype
can be found in base_config_ocn_only.yaml. Different runs, named as R*, where *
is a numeric character, build upon the basic configuration and override some of
the configuration parameters.

The code has recently been completely reqritten to allow overriding the base
configuration using yaml files. This is a much neater and more modular version
of the code and easier to maintain. In the new code design, all python files are
only written once and use configuration files to describe the emulator. Below
are the description of each file:
emulator.py: The main emulator file that contains OcnTrainer, OcnPreprocessor,
OcnPreprocessed, and OcnEvaluator. 
train.py: The training script
inference.py: The inference script

# Disclaimer
The preprocessing script is not wired until now. This needs some modifications
to adapt to the new design.

# Train
To train, simply use, for e.g.: 

python train.py --prototype R1 --dt 6h

In the above example, the prototype flag is required, but the dt flag is 
optional. It is only used to grab a bad samples yaml file from the prototype
home directory when an explicit bad_samples.yaml is not present in the run
directory (R1, here).
The inference can ve submitted only using 

python train.py --prototype R1

Support for TensorBoard is also added for logging and visualization of the 
training but this needs some more work for better results. 
 
