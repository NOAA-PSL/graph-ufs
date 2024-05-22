import logging
import os
import sys

from graphufs import DataGenerator, init_model, init_devices
from graphufs.torch import Dataset as TorchDataset
from p1 import P1Emulator

from ufs2arco import Timer

#from dask.cache import Cache
#
#cache = Cache(10e9)
#cache.register()

def pull_the_data(tds: TorchDataset):

    # note this is bad
    tds.xds.load()

    chunks = {
        "sample": 1,
        "lat": -1,
        "lon": -1,
        "channels": 13,
    }
    x, y = tds.get_xsample(0)
    inputs = tds._make_container(x, name="inputs", chunks=chunks)
    targets = tds._make_container(y, name="targets", chunks=chunks)

    inputs.to_zarr(tds.local_inputs_path, compute=False, mode="w")
    targets.to_zarr(tds.local_targets_path, compute=False, mode="w")

    for idx in range(len(tds)):
        tds._store_sample(idx, chunks=chunks)
        if idx % 10 == 0:
            logging.info(f"Done with {idx}")


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    timer = Timer()

    # parse arguments
    # 1. This sets normalization and stacked_normalization
    p1, args = P1Emulator.from_parser()

    # 2. Pull the training and validation data and store to data/data.zarr
    for mode in ["training", "validation"]:
        logging.info(f"Downloading {mode} data")
        tds = TorchDataset(p1, mode=mode)

        timer.start()
        pull_the_data(tds)
        timer.stop()

    logging.info("Done preprocessing")
