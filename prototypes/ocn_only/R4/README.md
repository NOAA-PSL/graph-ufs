# Ocn-only run R4
This is similar to run R3, i.e., with one IC and reorganized vertical levels, 
but with changes in hyperparameters.
Below hyperparameter changes were tested:
1. [T1] Change gnn_msg_steps to 8 (from 16) to see if this improves the 
   overfitting seen in R3.
   
   [T1x] This trained for 3 epochs in the debug queue, and the verdict is that 
   it indeed reduced the overfitting as evidenced from the loss curves. Below 
   are the take aways:
   - Training and validation loss track closely:
     * There's no significant divergence in 3 epochs of training, which is 
       great. 
     * This indicates reduced overfitting compared to the earlier run with 16 
       message steps.
   - Validation loss decreases consistently:
     * A downward trend across the 3 epochs suggests the model is learning and 
       generalizing reasonably well so far.
   - Learning rate annealing is smooth:
     * The cosine decay is behaving as expected, meaning no large jumps or 
       flatlining, and no instability signs.
   - Inferences ran on the third model check point showed relatively poor 
     performance compared to the pre-existing plots on the notebook. They were 
     from 192 neurons run I think.

2. [T2] Previously, in R3, we saw that reducing the latent size to 192 helped 
   with overfitting -- the training and validation losses were going down -- but
   the generalization was still an issue. In fact, it looked like it is 
   suffering from underfitting and not generalizing well with the validation 
   data even early on. So, we should instead focus on tuning this model to get 
   overlapping training/validation loss with a downward trend. So, here we are 
   experimenting with gnn_msg_steps=12 and latent_size=256 to find a common 
   ground. 
   
   [T2x] Here are the takeaways from the loss plot:
   - Training is stable: 
     * No NaNs or spikes — smooth learning behavior throughout all epochs.
   - Validation loss decreases early and steadily: 
     * Indicates that the model is effectively learning generalizable patterns, 
       not just memorizing.
   - Train–validation loss gap is smaller than in the 8-step case: 
     * Suggests that increasing GNN depth from 8 to 12 helped restore some 
       generalization capacity.
   
   Things to Watch
   - A slight growing gap is visible after epoch ~3.5: 
     * Training loss continues to improve. 
     * Validation loss improves more slowly or flattens. This may evolve into 
       overfitting if training continues without early stopping.
   - Peak learning rate (1e-3) may still be a bit high for this depth:
     * Not necessarily bad, but the training  might benefit from lowering to 
       3e-4 to get smoother convergence at later epochs. Alternatively, increase 
       cosine decay speed to compress the LR schedule.

3. [T3] This is T2, i.e., gnn_msg_steps=12 and latent_size=256, but with peak 
   learning rate equal to 1e-4.
   
   [T3x] This is T3 experess trained on the debug queue -- whatver the number of 
   epochs it could accommodate. Below are the takeaways:
   - Extremely stable training:
     * Loss curves are perfectly smooth and monotonically decreasing with the 
       number of epochs.
     * No NaNs, spikes, or oscillations — very healthy numerics.
   - Tight alignment between training and validation loss:
     * The gap is minimal throughout 4 epochs.
     * This indicates excellent generalization — the model isn't just fitting 
       the training set, it's actually learning transferable features.
   - The inferences are showing significant error growth near boundaries in SSH 
     for 5-10 days lead time. 
     * This indicates that nodes closer to the boundaries are getting more 
       sparse information from its neighbours which may themselves be completely 
       uninformed. The errors may compound over autoregressive lead times and 
       lead to this enhanced error buildup around boundaries. The possible 
       remedies are:
       + Use more gnn message passing steps and use small noise in the inputs 
     	 (or dropout) if it leads to overfitting.
       + Use bathymetry, which may help with this issue.
       + Use a boundary-aware mask in the loss function.
       + Let the model "know" that a node is near a boundary by encoding, say, 
         distance to land, or coastal proximity
             
   [T3M] With wrong statistics:
   Trained up to 40 epochs. Below are the takeaways:
   What’s Working Well:
   - Training and validation losses are nearly parallel throughout all 40 
     epochs:
     * No widening gap/divergence between the two.
     * Suggests strong generalization and no overfitting, even over long 
       training.
   - Validation loss keeps improving steadily up to the final epoch:
     * This indicates that the low learning rate is effective even on the 
       longer training
     * No flattening or early stagnation.
   - Training is highly stable:
     * No spikes, loss explosions, or instability even at latter epochs
     * Further confirms that 1e-4 peak LR is safe and reliable for the 
       architecture

   What may need improvement:
   - Inferences showing error build up around boundaries:
     * Looking at SSH inferences for 5-10 days lead times, it seems like error 
       is building up along the boundaries for longer autoregressive rollouts.
     * This is same for inferences from both epoch 26 and epoch 40 (even the 
       overall spatial RMSE looks similar for both epochs).    
      - May need bathymetry for better performance.
   - The training is looking fine but not steller. 
     * The training can perhaps be further improved by adding 10% noise in the 
       inputs.
     * I'm not too confident but maybe gnn_msg_passing_steps=16 can improve 
       things. This is not guaranteed though.
    
   [T3M] With corrected statistics:
   Trained for the same 40 epochs. Below is my naked eye analysis:
   - Training and validation loss are not too different 
     * Compared to the above training where I used wrong statistics, the
       training and validation lossoutcomes did not differ much.
     * Squinting more suggested that the noise amplitudes in loss per optim
       step are smaller in the new training compared to the previous one.
     * Also looks like the validation loss per epoch did not saturate as early
       as seen in the previous training.      
   - Inferences are too different as well.
     * I can make a more cleaner comparison of inference errors as a function
       of the lead time, but I'm sure there is not much difference.

   [T4] Same as T3 but with gaussian noise added to inputs with amplitude as 10% 
   of the corresponding standard deviation. Note that gaussian noise is not 
   added to the land-sea mask and time encodings, i.e., landsea_mask, 
   land_static, day_progress_cos, day_progress_sin, year_progress_cos, 
   year_progress_sin. However, because no masking was applied to the noise 
   fields, they had non-zero values over land. This is not consistent and, 
   ideally, the noise should not be applied outside of a variable's domain. This 
   was corrected. 
   
   [T4x] T4 submitted to the debug queue for training.     
   The loss curve for 3 epochs shows that 
   - The training is stable. 
     * Although the model starts with a very bad initial parameter values, 
       leading to exceptionally high loss value, it imroves continuously and the 
       loss decreases by 2 orders of magnitude in less than 3 epochs. This is 
       promising and shows that neural network is quickly learning to filter 
       noise.  
   - Validation dataset with and without noise:
     * The one where validation dataset inputs also has noise shows close 
       alignment between training and validation loss. When the validation loss 
       doesn't have any noise, the validation loss is much lower than the 
       training loss per epoch. This suggests that even model trained using 
       noisy inputs has utility for non-noisy inputs and model is quickly 
       learning to get rid of this noise.
   - Comparing the loss curve to the case where no masking in noise was applied, 
     I did not see a huge difference between the two loss curves. 
     * This was probably due to the fact that masking is applied in the loss and 
       therefore the loss corresponds to the actual domain for each field 
       irrespective whether masking is applied to noise or not. Masking the 
       noise however is very likely to guide neural networks in the right 
       direction.
   - It makes sense to train this longer, maybe for 40 epochs like before. 
   
   [T4M] Trained the T4 prototype for full 40 epochs. Below are the key
   takeaways. 
   - Over 40 epochs, the training loss continues to improve but the validation
     loss starts increasing -- signalling a textbook example of overfitting.
   - The above means that at 10% noise level, the network started learning
     noise.
   - This is despite a lower peak LR(= 1e-4), the same weight decay (=0.1), and
     lower gnn_msg_passing_steps(=12).

   [T5] Same as T4 but noise amplitude is set to 1% of the std of each channel.
   [T5x] T5 submitted to the debug queue.
  
