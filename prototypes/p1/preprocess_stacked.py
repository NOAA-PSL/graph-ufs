import logging
import os
import sys

from graphufs import DataGenerator, init_model, init_devices
from graphufs.torch import Dataset as TorchDataset
from p1 import P1Emulator

from ufs2arco import Timer

from dask.cache import Cache

cache = Cache(10e9)
cache.register()

def pull_the_data(tds: TorchDataset):

    chunks = {
        "sample": 1,
        "lat": -1,
        "lon": -1,
        "channels": 13,
    }
    x, y = tds.get_xsample(0)
    inputs = tds._make_container(x, name="inputs", chunks=chunks)
    targets = tds._make_container(y, name="targets", chunks=chunks)

    inputs.to_zarr(f"{tds.emulator.local_store_path}/{tds.mode}/inputs.zarr", compute=False, mode="w")
    targets.to_zarr(f"{tds.emulator.local_store_path}/{tds.mode}targets.zarr", compute=False, mode="w")

    for idx in range(len(tds)):
        tds._store_sample(idx, chunks=chunks)
        if idx % 10 == 0:
            print(f"Done with {idx}")


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    timer = Timer()

    # parse arguments
    p1, args = P1Emulator.from_parser()

    # 1. Read remote normalization, store locally, and set to p1
    p1.set_normalization()

    # 2. Pull the training and validation data and store to data/data.zarr
    logging.info("Downloading Training Data")
    training_data = TorchDataset(p1, mode="training")

    timer.start()
    pull_the_data(training_data)
    timer.stop()
    logging.info("Done preprocessing")
