#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    _expt = "R8"
    _subexpt = "T6M"
    
    ds = xr.load_dataset(f"/pscratch/sd/n/nagarwal/ocn-only/{_expt}/loss.nc")

    fig, axs = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)

    axLR = axs[0].twinx()
    l1 = ds.loss.plot(ax=axs[0], color="C0", label="Training Loss",)
    l2 = ds.learning_rate.plot(ax=axLR, color="gray", label="Learning Rate")

    ds.loss_train.plot(ax=axs[1], label="Training",)
    ds.loss_valid.plot(ax=axs[1], label="Validation",)

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)
    axLR.spines["top"].set_visible(False)

    # labels and stuff
    axs[0].set(
        xlabel="Optimization Step",
        ylabel="Loss Value",
        yscale="log",
    )
    axLR.set(ylabel="Learning Rate")
    axs[1].set(
        xlabel="Epoch",
        ylabel="Loss Value",
        yscale="log",
    )
    lines = [l1[0], l2[0]]
    axs[0].legend(
        lines,
        list(l.get_label() for l in lines),
        loc="center right",
    )
    axs[1].legend()

    fig.savefig(f"figures/training_loss_{_expt}_{_subexpt}.png", dpi=300)
