#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Experiments
    expts = ["T1M","T3M", "T2M", "T4M"]
    noise_amp =[0, 0.1, 1, 5] 
    
    expt_dir = "/pscratch/sd/n/nagarwal/ocn-only/R2/"

    # Initialize subplots
    fig, axs = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)

    axLR = axs[0].twinx()

    # Loop over the experiments
    for i, expt in enumerate(expts):
        ds = xr.load_dataset(f"{expt_dir}/loss_{expt}.nc")
        
        # left panel
        l1 = ds.loss.plot(ax=axs[0], color=f"C{i}", label=f"Training Loss - {noise_amp[i]}% noise",)
   
        # right panel
        ds.loss_train.plot(ax=axs[1], color=f"C{i}", linestyle="-", label=f"Training - {noise_amp[i]}% noise",)
        ds.loss_valid.plot(ax=axs[1], color=f"C{i}", linestyle="--", label=f"Validation - {noise_amp[i]}% noise",)  	
   
    l2 = ds.learning_rate.plot(ax=axLR, color="gray", label="Learning Rate")

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)
    axLR.spines["top"].set_visible(False)

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    # labels and stuff
    axs[0].set(
        xlabel="Optimization Step",
        ylabel="Loss Value",
    )
    axLR.set(ylabel="Learning Rate")
    axs[1].set(
        xlabel="Epoch",
        ylabel="Loss Value",
    )
    #lines = [l1[0], l2[0]]
    #axs[0].legend(
    #    lines,
    #    list(l.get_label() for l in lines),
    #    loc="center right",
    #)
    axs[0].legend(loc="center right")
    axs[1].legend(loc="upper right")

    fig.savefig("figures/training_loss_R2_multiple_noise_amplitudes.jpeg", dpi=300)
