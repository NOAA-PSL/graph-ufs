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

# Preprocessing
To submit preprocessing, use
 
python preprocess.py --prototype R1

in this directory, not in the prototype/run directory. Unlike the legacy code
design, this one would place submit_preprocess.sh in the job_scripts directory 
in this folder. This can potentially be changed but I don't think this is super 
important. Just make sure that you are firing the above command from this ocn_only 
folder and not from an R*/job_scripts folder. 
Note that although the preproocess script accepts another argument, dt, but this
is more of a requirement for training, not preprocessor.

# Statistics
The script to generate the statistics has also been wired. To compute statistics
for a given run, use

python calc_statistics.py --prototype R2 --comp ocn --spatial_avg False

The comp flag corresponds to the component for which you want to compute the
statistics. This can take values: ocn/atm/land/ice
The spatial_avg flag determines whether spatially dependent or spatially
averaged statistics are computed. This is an optional arguement with the default
being True.

Note that there is separate emulator class for statistics defined in
emulator_for_statistics.py, which has a lot of the same code as in emulator.py.
This means there is a lot of code duplication going on here.
A potential way to avoid this code duplication is by deriving the
OcnTrainer from the statistics class where all of the OcnTrainer class contents
would be there except the last line. So the OcnTrainer class would only have an
extra line corresponding to the superclass constructor call. 

Another thing worth noting is that firing the statistics computation would lead
a lot of jobs in the debug queue. An efficient strategy for computing is the
statistics is to just fire 3D variables on the debug queue and compute the
statistics for the 2D variables in an interactive job by directly firing the
python command in the job script, for e.g., 

python -c 'from calc_statistics import main ; main("R10", "spfh2m", "atm", True)'

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
training.

# Configurations:
* R0: 6h emulator, 2 IC, fineTop
* R1: 24h emulator, 2 IC, fineTop
* R2: 6h emulator, 2 IC, coarseTop
* R3: 6h emulator, 1 IC, coarseTop 
* R4: 6h emulator, 1 IC, coarseTop, hyperparams tuning. T3M: best
* R5: 24h emulator, 1 IC, coarseTop
* R6: 24h emulator, 2 IC, coarseTop
* R7: 24h emulator, 1 IC, coarseTop with bathymetry
* R8: 24h emulator, 1 IC, coarseTop with bathymetry and using mahalanobis loss
* R9: 6h emulator, 1 IC, coarseTop with Bathymetry 
* R10: 24h emulator, 1 IC, fineTop (no bathy) and using MSE loss
* R11: 6h emulator, 1 IC, fineTop (no bathy) and using MSE loss
