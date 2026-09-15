#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Compares the global-covariance baseline (R8/T13M) against the
    # location-based-covariance run (R12/T1M). R12 only reuses R8's
    # preprocessed input data (see R12/config.yaml's local_store_path) --
    # its own model/loss/tensorboard outputs live under R12's output_dir,
    # so each experiment needs its own directory here.
    expts = [("R8", "T13M"), ("R12", "T1M")]
    expt_name = ["MLoss-global-cov", "MLoss-space-dependent-cov"]

    # Initialize subplots
    fig, axs = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)

    axLR = axs[0].twinx()

    # Loop over the experiments
    for i, (expt_dir, expt) in enumerate(expts):
        ds = xr.load_dataset(f"/pscratch/sd/n/nagarwal/ocn-only/{expt_dir}/loss_{expt}.nc")
        
        # left panel
        l1 = ds.loss.plot(ax=axs[0], color=f"C{i}", 
            label=f"Training Loss - {expt_name[i]}",)
        # right panel
        ds.loss_train.plot(ax=axs[1], color=f"C{i}", linestyle="-",
            label=f"Training - {expt_name[i]}",)
        ds.loss_valid.plot(ax=axs[1], color=f"C{i}", linestyle="--",
            label=f"Validation - {expt_name[i]}",)  	
   
    l2 = ds.learning_rate.plot(ax=axLR, color="gray", label="Learning Rate")

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)
    axLR.spines["top"].set_visible(False)

    #axs[0].set_yscale("log")
    #axs[1].set_yscale("log")

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
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    fig.savefig("figures/training_loss_R8_T13M_vs_R12_T1M.jpeg", dpi=300)
