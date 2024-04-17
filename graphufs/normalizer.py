import os
import numpy as np
import xarray as xr
from ufs2arco.timer import Timer


class Normalizer():
    """class for computing normalization statistics:
    * mean_by_level
    * stddev_by_level
    * diffs_stddev_by_level
    """

    path_in = None      # original dataset
    path_out = None     # path to save normalization statistics
    start_date = None   # start date to subsample data
    end_date = None     # end date to subsample data, inclusive
    time_skip = None    # integer used to skip in time
    open_zarr_kwargs = None
    to_zarr_kwargs = None
    load_full_dataset = False

    def __init__(
        self,
        path_in: str,
        path_out: str,
        start_date: str = None,
        end_date: str = None,
        time_skip: int = None,
        open_zarr_kwargs: dict = None,
        to_zarr_kwargs: dict = None,
        load_full_dataset: bool = False,
        ):

        self.path_in = path_in
        self.path_out = path_out
        self.start_date = start_date
        self.end_date = end_date
        self.time_skip = time_skip
        self.open_zarr_kwargs = open_zarr_kwargs if open_zarr_kwargs is not None else dict()
        self.to_zarr_kwargs = to_zarr_kwargs if to_zarr_kwargs is not None else dict()
        self.load_full_dataset = load_full_dataset

        self.delta_t = f"{self.time_skip*3} hour" if self.time_skip is not None else "3 hour"


    def __call__(self, data_vars=None):

        walltime = Timer()
        localtime = Timer()
        walltime.start()

        localtime.start("Setup")
        ds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)

        # select variables
        if data_vars is not None:
            if isinstance(data_vars, str):
                data_vars = [data_vars]
            ds = ds[data_vars]

        # subsample in time
        ds = self.subsample_time(ds)
        localtime.stop()

        # load if not 3D
        if self.load_full_dataset:
            localtime.start("Loading the whole dataset...")
            ds = ds.load();
            localtime.stop()

        # do the computations
        localtime.start("Computing mean")
        self.calc_mean_by_level(ds)
        localtime.stop()

        localtime.start("Computing stddev")
        self.calc_stddev_by_level(ds)
        localtime.stop()

        localtime.start("Computing diff stddev")
        self.calc_diffs_stddev_by_level(ds)
        localtime.stop()

        walltime.stop("Total Walltime")


    def subsample_time(self, xds):
        """select time period and frequency we want"""
        with xr.set_options(keep_attrs=True):
            rds = xds.sel(time=slice(self.start_date, self.end_date))
            rds = rds.isel(time=slice(None, None, self.time_skip))
        return rds


    def calc_diffs_stddev_by_level(self, xds):
        """compute standard deviation of differences by level, store to zarr"""

        with xr.set_options(keep_attrs=True):
            result = xds.diff("time")
            result = result.std(["grid_xt", "grid_yt", "time"])

        for key in result.data_vars:
            result[key].attrs["description"] = f"standard deviation of temporal {self.delta_t} difference over lat, lon, time"
            result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

        this_path_out = os.path.join(
            self.path_out,
            "diffs_stddev_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        return result


    def calc_stddev_by_level(self, xds):
        """compute standard deviation by level, store to zarr"""

        with xr.set_options(keep_attrs=True):
            result = xds.std(["grid_xt", "grid_yt", "time"])

        for key in result.data_vars:
            result[key].attrs["description"] = "standard deviation over lat, lon, time"
            result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

        this_path_out = os.path.join(
            self.path_out,
            "stddev_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        return result


    def calc_mean_by_level(self, xds):
        """compute mean by level, store to zarr"""

        with xr.set_options(keep_attrs=True):
            result = xds.mean(["grid_xt", "grid_yt", "time"])

        for key in result.data_vars:
            result[key].attrs["description"] = "average over lat, lon, time"
            result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

        this_path_out = os.path.join(
            self.path_out,
            "mean_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        return result


    @staticmethod
    def _time2str(xval):
        """turn an xarray numpy.datetime64 -> string at hourly rep"""
        return str(xval.values.astype("M8[h]"))
