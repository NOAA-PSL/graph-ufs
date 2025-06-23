# Ocn-only run
In this run, we are increasing the time step of the ocn-only model to 24hr as
opposed to 6hr used in R0. This is after numerous failed attempts to emulate 6hr
increments in the ocean. The hypothesis is that there is not enough signal to
learn in such a short time step and therefore the model is overfitting.
Presumabely, 24hr increments would have more coherent signals to learn and
emulate.
 
## Outcome
- Increasing the delta_t_model only did not help a ton.
