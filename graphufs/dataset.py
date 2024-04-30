import xarray as xr

from torch.utils.data import Dataset as TorchDataset

from xbatcher import BatchGenerator

from graphcast.data_utils import extract_inputs_targets_forcings
from graphcast.model_utils import dataset_to_stacked
from graphcast.xarray_jax import unwrap

class GraphUFSDataset(TorchDataset):

    @property
    def xds(self):
        return self.sample_generator.ds

    def __init__(
        self,
        emulator,
        mode,
    ):

        self.emulator = emulator
        self.mode = mode
        xds = self._open_dataset()
        self.sample_generator = BatchGenerator(
            ds=xds,
            input_dims={
                "datetime": emulator.n_forecast,
                "lon": len(xds["lon"]),
                "lat": len(xds["lat"]),
                "level": len(xds["level"]),
            },
            # adding batch_dims does nothing
            input_overlap={
                "time": emulator.n_input,
            },
            preload_batch=False,
        )


    def __len__(self):
        return len(self.sample_generator)


    def __getitem__(self, idx): # returns dict with inputs targets forcings
        sample_input, sample_target, sample_forcing = self.get_xarrays(idx)

        X = self._stack(sample_input, sample_forcing)
        y = self._stack(sample_target)
        return X, y

    @staticmethod
    def _xstack(a, b=None):
        """results in array with shape [lat, lon, batch, channels] which is the easiest shape to work with
        other than a totally flat array, because the StackedGraphCast has to reshape everything
        from a flat array, and this lat_lon_leading shape requires no assumptions regarding number of dimensions

        Returns:
            result (xarray.DataArray):
        """
        result = dataset_to_stacked(a)
        if b is not None:
            result = xr.concat(
                [result, dataset_to_stacked(b)],
                dim="channels",
            )
        result = result.transpose("batch", "lat", "lon", "channels")
        return result

    def _stack(self, a, b=None):
        """same thing as xstack, but unpack data from xarray

        Returns:
            result (chex.Array)
        """
        xresult = self._xstack(a, b)
        return xresult.data.squeeze()

    def _open_dataset(self):

        xds = self.emulator.open_dataset()
        time = self.emulator.get_time(mode=self.mode)
        xds = self.emulator.subsample_dataset(xds, new_time=time)
        xds = xds.rename({
            "time": "datetime",
            "pfull": "level",
            "grid_yt": "lat",
            "grid_xt": "lon",
        })
        xds = xds.drop_vars(["cftime", "ftime"])
        return xds


    def _preprocess(self, xds):
        xds["time"] = xds["datetime"] - xds["datetime"][0]
        xds = xds.swap_dims({"datetime": "time"}).reset_coords()
        xds = xds.set_coords(["datetime"])
        return xds

    def get_xda(self, idx : int):
        sample = self.sample_generator[idx]
        sample = self._preprocess(sample)
        sample = sample.load()
        return sample

    def get_xarrays(self, idx : int):
        sample = self.get_xda(idx)

        sample_input, sample_target, sample_forcing = extract_inputs_targets_forcings(
            sample,
            **self.emulator.extract_kwargs,
        )
        sample_input = sample_input.expand_dims({"batch": [idx]})
        sample_target = sample_target.expand_dims({"batch": [idx]})
        sample_forcing = sample_forcing.expand_dims({"batch": [idx]})
        return sample_input, sample_target, sample_forcing

    def get_batch_of_xarrays(self, indices : list[int]):
        xinputs = []
        xtargets = []
        xforcings = []
        for idx in indices:
            xi, xt, xf = self.get_xarrays(idx)
            xinputs.append(xi)
            xtargets.append(xt)
            xforcings.append(xf)

        xinputs = xr.concat(xinputs, dim="batch")
        xtargets = xr.concat(xtargets, dim="batch")
        xforcings = xr.concat(xforcings, dim="batch")
        return xinputs, xtargets, xforcings


    def get_xsample(self, idx : int):
        sample_input, sample_target, sample_forcing = self.get_xarrays(idx)

        X = self._xstack(sample_input, sample_forcing)
        y = self._xstack(sample_target)
        return X, y
