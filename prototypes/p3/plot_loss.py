#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import graphufs

if __name__ == "__main__":
    plt.style.use("graphufs.plotstyle")

    dsdict = {
        key: xr.load_dataset(f"/pscratch/sd/t/timothys/p3/{key}/loss.nc")
        for key in ["uvwc", "uvnc", "uvncbs32", "nvnc"]
    }

    fig, axs = plt.subplots(1, 2, figsize=(8,4), constrained_layout=True, sharey=True)

    for ykey, label, ax in zip(["loss_avg", "loss_valid"], ["Training Loss", "Validation Loss"], axs):
        for key, xds in dsdict.items():
            xds[ykey].plot(ax=ax, label=key)

        ax.set(
            xlabel="Epoch",
            ylabel=label,
            title=label,
        )
        ax.legend()

    fig.savefig("figures/loss.pdf")
