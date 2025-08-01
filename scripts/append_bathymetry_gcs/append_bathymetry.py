"""Compute static variables surface orography and land/sea mask,
append it back to the original or store it locally depending on inputs

Note: this was heavily borrowed from this xarray-beam example:
https://github.com/google/xarray-beam/blob/main/examples/era5_climatology.py
"""

from typing import Tuple
import sys
from absl import app
from absl import flags
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.dask.dask_runner import DaskRunner
import numpy as np
import xarray as xr
import xarray_beam as xbeam

from ufs2arco import Layers2Pressure
from localzarr import ChunksToZarr


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', "DirectRunner", 'beam.runners.Runner')
NUM_WORKERS = flags.DEFINE_integer('num_workers', None, help="Number of workers for the runner")
NUM_THREADS = flags.DEFINE_integer('num_threads', None, help="Passed to ChunksToZarr")

def return_bathymetry(
    key: xbeam.Key,
    xds: xr.Dataset,
) -> Tuple[xbeam.Key, xr.Dataset]:
    """Return dataset with the batymetry field, that's it"""

    newds = xr.Dataset()
    bathymetry = xds["depth"] if "time" not in xds["depth"] else xds["depth"].isel(time=0)

    newds["depth"] = bathymetry
    newds["depth"].attrs = xds["depth"].attrs.copy()

    for k in ["time", "cftime", "ftime", "pfull"]:
        if k in newds:
            newds = newds.drop_vars(k)
    return key, newds

def main(argv):

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,)

    path = INPUT_PATH.value
    kwargs = {}

    if "gs://" in path or "gcs://" in path:
        kwargs["storage_options"] = {"token": "anon"}

    source_dataset, source_chunks = xbeam.open_zarr(path, **kwargs)
    source_dataset = source_dataset[["depth",]].isel(time=0) if "time" in source_dataset.dims else source_dataset[["depth",]]
    for key in ["time", "cftime", "ftime", "pfull"]:
        if key in source_dataset:
            source_dataset = source_dataset.drop_vars(key)
        if key in source_chunks:
            source_chunks.pop(key)

    # create template
    _, tds = return_bathymetry(None, source_dataset)
    #input_chunks = source_chunks.copy()
    output_chunks = {k: v for k,v in source_chunks.items() if k not in ("pfull", "z_l", "time")}
    input_chunks=output_chunks.copy()

    template = xbeam.make_template(tds)
    storage_options = None
    if "gs://" in OUTPUT_PATH.value:
        storage_options = {"token": "/global/homes/n/nagarwal/.gcs/replay-service-account.json"}

    pipeline_kwargs = {}
    if NUM_WORKERS.value is not None:
        pipeline_kwargs["options"]=PipelineOptions(
            direct_num_workers=NUM_WORKERS.value,
        )

    with beam.Pipeline(runner=RUNNER.value, argv=argv, **pipeline_kwargs) as root:
        (
            root
            | xbeam.DatasetToChunks(source_dataset, input_chunks, num_threads=NUM_THREADS.value)
            | beam.MapTuple(return_bathymetry)
            | ChunksToZarr(OUTPUT_PATH.value, template, output_chunks, num_threads=NUM_THREADS.value, storage_options=storage_options)
        )

    logging.info("Done")

if __name__ == "__main__":
    app.run(main)
