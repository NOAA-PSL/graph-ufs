import numpy as np
import xarray as xr
from timer import Timer


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

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                getattr(key)
            except:
                raise KeyError(f"Normalizer.__init__: can't find attr {key}")

            setattr(self, key, val)

        self.delta_t = f"{self.time_skip*3} hour"


    def __call__(self, data_vars=None):

        ds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)

        # select variables
        if data_vars is not None:
            if isinstance(data_vars, str):
                data_vars = [data_vars]
            ds = ds[data_vars]

        # subsample in time
        ds = self.subsample_time(ds)

        # do the computations
        walltime = Timer()
        localtime = Timer()

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
            result[key].attrs["start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["end_date"] = self._time2str(xds["time"][-1])

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
            result[key].attrs["start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["end_date"] = self._time2str(xds["time"][-1])

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
            result[key].attrs["start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["end_date"] = self._time2str(xds["time"][-1])

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
