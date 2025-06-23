# Atm-only run
This is the first attempt to run the atm + surface ocean configuration + land + ice 
with full 27 years of REPLAY data. For land, I have removed soil temperature due to 
unnecessary data outside the land mask. This will corrupt the statistics and hence the
training after applying the land mask to it. Not applying mask during training was one 
possible solution, but then this will give more weightage to sst, if used in the ocean
cells of soilt1, which is undesired.

# Outcome
