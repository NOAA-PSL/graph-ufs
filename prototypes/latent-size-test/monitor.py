#!/usr/bin/env python
# coding: utf-8

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import graphufs

def plot_single_experiment(xds):

    fig, axs = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)

    axLR = axs[0].twinx()
    l1 = xds.loss.plot(ax=axs[0], color="C0", label="Training Loss")
    l2 = xds.learning_rate.plot(ax=axLR, color="C2", label="Learning Rate")

    xds.loss_avg.plot(ax=axs[1], label="Training")
    xds.loss_valid.plot(ax=axs[1], label="Validation")

    for ax in axs:
        for key in ["right", "top"]:
            ax.spines[key].set_visible(False)
    axLR.spines["top"].set_visible(False)

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
    lines = [l1[0], l2[0]]
    axs[0].legend(
        lines,
        list(l.get_label() for l in lines),
        loc="center right",
    )
    axs[1].legend()

    savedir = f"results/ls{xds.latent_size.values[0]:03d}"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    fig.savefig(f"{savedir}/loss.pdf")
    return

def plot_all(xds):

    fig, axs = plt.subplots(1, 2, figsize=(8,4), constrained_layout=True)

    for y, label, ax in zip(["loss_avg", "loss_valid"], ["Training Loss", "Validation Loss"], axs):
        xds[y].plot.line(ax=ax, x="epoch")
        ax.set(title=label, xlabel="Epochs", ylabel="Loss Averaged Over Epoch")
    fig.savefig(f"results/combined_loss.pdf")
    return


if __name__ == "__main__":

    plt.style.use("graphufs.plotstyle")

    dslist = []

    for latent_size in [16, 64, 128, 256]:
        xds = xr.load_dataset(f"/testlfs/latent-size-test-{latent_size:03d}/loss.nc")
        xds = xds.expand_dims({"latent_size": [latent_size]})
        plot_single_experiment(xds)

        dslist.append(xds)

    ds = xr.concat(dslist, dim="latent_size")
    plot_all(ds)


