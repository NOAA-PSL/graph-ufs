import logging
import os
import sys

from graphufs import DataGenerator, init_model, init_devices
from p1 import P1Emulator


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    p1, args = P1Emulator.from_parser()
    init_devices(p1)

    # 1. Read remote normalization, store locally, and set to p1
    p1.set_normalization()

    # 2. Pull the training and validation data and store to data/data.zarr
    logging.info("Downloading Training Data")
    p1.get_the_data(mode="training")
    logging.info("Downloading Validation Data")
    p1.get_the_data(mode="validation")

    # 3. Preprocessing, make sure to pass --chunks-per-epoch arg
    logging.info("Preprocessing")

    gen_train = p1.get_batches(mode="training")
    for i in range(p1.chunks_per_epoch):
        logging.info(f"Processing training chunk {i}")
        next(gen_train)

    gen_valid = p1.get_batches(mode="validation")
    for i in range(p1.chunks_per_epoch):
        logging.info(f"Processing validation chunk {i}")
        next(gen_valid)

    logging.info("Done with preprocessing")
