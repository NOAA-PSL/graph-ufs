# Ocn-only run
This is similar to the R1 attempt, i.e., run the ocn-only config with full REPLAY data and
with the indices of bad samples supplied to omit them during the training, but we work
with slighly less complex model (graphcast) in this run. In particular, we are interested 
in utilizing a smaller hierarchy of multimesh to avoid the overfitting issue seen in R1. 
This will presumabely not allow the model to learn grid scale noise which appear in the 
autogregressive rollout and gradually worsens the forecast.

## Outcome
