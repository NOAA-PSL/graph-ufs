import os
import logging
import time
import dask
import shutil

from graphufs import (
    predict,
    DataGenerator,
    init_devices,
    convert_wb2_format,
    compute_rmse_bias,
)

from p1 import P1Emulator
from ufs2arco import Timer

if __name__ == "__main__":

    timer1 = Timer()

    # parse arguments
    p1, args = P1Emulator.from_parser()

    # for multi-gpu training
    init_devices(p1)

    # configuring dask
    if p1.dask_threads is not None:
        dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    # data generators
    testor = DataGenerator(
        emulator=p1,
        n_optim_steps=p1.steps_per_chunk,
        max_queue_size=p1.max_queue_size,
        num_workers=p1.num_workers,
        mode="testing",
    )

    # load weights
    logging.info(f"Loading weights: {args.id}")
    params, state = p1.load_checkpoint(args.id)

    logging.info("Starting Testing")

    # create predictions and targets zarr file for WB2
    predictions_zarr_name = f"{p1.local_store_path}/graphufs_predictions.zarr"
    targets_zarr_name = f"{p1.local_store_path}/graphufs_targets.zarr"
    if os.path.exists(predictions_zarr_name):
        shutil.rmtree(predictions_zarr_name)
    if os.path.exists(targets_zarr_name):
        shutil.rmtree(targets_zarr_name)

    stats = {}
    for c in range(p1.chunks_per_epoch):
        logging.info(f"Testing on chunk {c+1}")
        start1 = time.time()

        # get chunk of data in parallel with inference
        data = testor.get_data()

        # run predictions
        predictions = predict(
            params=params,
            state=state,
            emulator=p1,
            testing_data=data,
        )

        targets = data["targets"]
        inittimes = data["inittimes"]

        # Compute rmse and bias comparing targets and predictions
        compute_rmse_bias(predictions, targets, stats, c)

        # write predictions chunk by chunk to avoid storing all of it in memory
        predictions = convert_wb2_format(p1, predictions, inittimes)
        predictions = predictions.dropna("time")
        predictions.to_zarr(predictions_zarr_name, append_dim="time" if c else None)

        # write also targets to compute metrics against it with wb2
        targets = convert_wb2_format(p1, targets, inittimes)
        targets = targets.dropna("time")
        targets.to_zarr(targets_zarr_name, append_dim="time" if c else None)

        end1 = time.time()
        logging.info(f"Done with chunk {c+1} in: {end1 - start1:.4f} sec")

    logging.info("--------- Statistiscs ---------")
    for k, v in stats.items():
        logging.info(f"{k:32s}: RMSE: {v[0]} BIAS: {v[1]}")

    # stop the data generators
    testor.stop()
    logging.info("Done Testing")
