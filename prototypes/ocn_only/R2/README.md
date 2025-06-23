# Ocn-only run
In this run, we have removed too many layers near the surface ocean as it was
possibly leading to ill-conditioned feature matrix and therefore the grid
artifacts we were seeing in the inferences. In the new setup, the vertical
levels are more uniformly spaced than before  and do not possess as much
vertical correlation as before.  

A brief literature review on the vertical
levels of ocean-only emulators suggested that the existing emulators do not
possess too much vertical resolution near the surface ocean -- as done in
physical ocean modeling. For e.g., GLONET possess only 3 layers in the top 100m
ocean. Although OLA probably had more vertical resolution around the surface but
full credibility of this model is yet to be cofirmed.  
 
## Outcome
