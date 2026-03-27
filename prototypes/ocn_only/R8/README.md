# Ocn-only R8
This is essentially a Mahalanobis loss based training configuration.
In this run, we consider a given configuration of the ocn-only emulator and
train it using mahalanobis loss. In the first experiment, although
including bathymtery did not result in any significant change in the results, I
believe it's important to include it as it may result in some positive changes
for a different configuration/architecture. All hyperparameters are kept the same.

# Running instructions
You can essentially run this prototype for any cofiguration developed so far by
changing the config.yaml in this directory and datasets (training/validation) in
the scratch directory. Additionally, consider changing the bad_samples.yaml
file, based on whether you are training a 6h/24h emulator with 1/2 ICs.
Alternatively, you can just delete this file altogther and pass dt using the
--dt flag while running the script.

# Configurations:
[T1M] A full-blown training of this prototype with 24h, 1 IC, coarseTop emulator
in the presence of bathymetry. Although including bathymtery did not result in 
any significant change in the results, I believe it's important to include it as 
it may result in some positive changes for a different configuration/architecture. 
All hyperparameters are kept the same.

[T2M] Training the 24h, 1IC, coarseTop emulator without bathymetry using the
mahalanobis loss. All hyperparams were the same.
This is the best 24h emulator trained with the mahalanobis loss, which makes it
even better. 

[T3M] Training 6h, 1IC, coarseTop emulator without bathy using the mahalanobis
loss. The training and validation data from the R4 prototype are transferred.

[T4M] Training the 24h, 1IC, coarseTop emulator without bathymetry using the
mahalanobis loss but using the tendency correlation matrix obtained with the
seasonality intact. Analysis has shown that the impact of seasonality is higher
on 24h tendency correlations than in 6h tendencies, especially on positive
correlations. It would be interesting to check how this impacts training and/or
inferences.

[T5M] Training the 6h, 1 IC, coarseTop emulator without bathy using the
mahalanobis loss but with the correlation matrix computed using the entire 27
years of training data and with seasonality removed. Previous analysis showed
that seasonality doesn't matter too much. Even if it doesn't make a huge
difference, I think it's important to use the one where we have used the whole
training data just like in the 24h emulator case.

[T6M] Training the 6h, 1 IC, coarseTop emulator with bathy using the mahalanobis
loss using the same correlation matrix used in T5M. 
This resulted in a very different training than its euclidean counterpart, in
the sense that the training and validation loss started with a very high value
(order of 10^6) but decreased gradually to order of 10 within 40 epochs. The
curve still seems to be going down and therefore longer training is required in
this case. Also the inferences for this prototype are blowing up I guess,
because it is producing an error saying "a mismatch between the grid2mesh_gnn
[63,256] and the input shape [62,256]", which is weird because everything is
correct. Also I think the model is able to produce forecasts for 1-2 rollouts
before throwing this error. So, I believe it's because of the training and
blowup of rollouts.

[T7M] Fired the above for 100 epochs of training. 
The same error as above occured during the inference. I'm wondering this is due
to the fact that I did not specify the new ocn input/targets/forcing due to the
inclusion of the bathymetry. Firing this training again with these values in the
config.yaml of R8.

[T8M] Indeed the above was true. It was due to the fact that I was not
specifying the new ocn inputs/targets/forcing due to the inclusion of
bathymetry. In this prototype, I did this and fired again and the both training
and inference went fine. Note that I forgot to change the sub_expt parameter in
the config.yaml and so it overwrote the T7M in  the tensorboard directory.
However this is fine coz T7M was crazy and wrong anyway.

[T2inf] This experiment was done to produce 6 months long lead time forecast
using the last model checkpoint for the first two time stamps of the validation
datetime. 
Moved the R5 dataset back to its directory.

[T9M] Training the 24h, 1IC, coarseTop configuration using the Mahalanobis loss
but with the space-dependent tendency covariance matrix. So this time the
covariance matrix has the size (192, 384, 29, 29) as opposed to only (29, 29)
matrix used earlier. It would be interesting to check how this works out. 

[T10M] Training 24h-2IC-coarseTop configuration using the Mahalanobis loss. Note
that the spatially averaged correlation matrix with the size (29, 29) is used
here. 

[T11M] Training 24h-1IC-fineTop configuration using the Mahalanobis loss with
spatially averaged correlation matrix.  
 
