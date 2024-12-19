#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import graphufs

if __name__ == "__main__":
    plt.style.use("graphufs.plotstyle")

    dsdict = {
        key: xr.load_dataset(f"/global/cfs/cdirs/m4718/timothys/p2p/{key}/loss.nc")
        for key in ["uvwc", "uvnc", "uvncbs32", "nvnc", "uvwcsic"]
    }

    fig, axs = plt.subplots(1, 2, figsize=(8,4), constrained_layout=True, sharey=True)

    for ykey, label, ax in zip(["loss_avg", "loss_valid"], ["Training Loss", "Validation Loss"], axs):
        for key, xds in dsdict.items():
            plotme = xds[ykey]
            if len(plotme) == 128:
                ax.plot(plotme.epoch / 2, plotme, label=key)
            else:
                plotme.plot(ax=ax, label=key)

        ax.set(
            xlabel="Epoch",
            ylabel=label,
            title=label,
        )
        ax.legend()

    fig.savefig("figures/loss.pdf")
