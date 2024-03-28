import os
import yaml
import warnings
import itertools
import dataclasses
import numpy as np
import pandas as pd
import xarray as xr
from jax import tree_util

from graphcast.graphcast import ModelConfig, TaskConfig
from graphcast.data_utils import extract_inputs_targets_forcings

class ReplayEmulator:
    """An emulator based on UFS Replay data. This manages all model configuration settings and normalization fields. Currently it is designed to be inherited for a specific use-case, and this could easily be generalized to read in settings via a configuration file (yaml, json, etc). Be sure to register any inherited class as a pytree for it to work with JAX.

    Example:
        see graph-ufs/prototypes/p0/simple_emulator.py for example usage
    """

    data_url = ""
    norm_urls = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }
    wb2_obs_url = ""
    
    local_store_path = None

    # these could be moved to a yaml file later
    # task config options
    input_variables = tuple()
    target_variables = tuple()
    forcing_variables = tuple()
    all_variables = tuple() # this is created in __init__
    pressure_levels = tuple()

    # time related
    delta_t = None              # the model time step
    input_duration = None       # time covered by initial condition(s)
    target_lead_time = None     # how long the forecast is, i.e., when we compare to data
    training_dates = tuple()    # bounds of training data (inclusive)
    testing_dates = tuple()     # bounds of testing data (inclusive)
    validation_dates = tuple()  # bounds of validation data (inclusive)

    # training protocol
    batch_size = None           # number of forecasts averaged over in loss per optim_step
    num_epochs = None           # number of epochs

    # model config options
    resolution = None
    mesh_size = None
    latent_size = None
    gnn_msg_steps = None
    hidden_layers = None
    radius_query_fraction_edge_length = None
    mesh2grid_edge_normalization_factor = None

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = None
    init_rng_seed = None
    training_batch_rng_seed = None # used to randomize the training batches

    # data chunking options
    chunks_per_epoch = None          # number of chunks per epoch
    steps_per_chunk = None           # number of steps to train for in each chunk
    checkpoint_chunks = None         # save model after this many chunks are processed

    def __init__(self):

        if self.local_store_path is None:
            warnings.warng("ReplayEmulator.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")

        pfull = self._get_replay_vertical_levels()
        levels = pfull.sel(
            pfull=list(self.pressure_levels),
            method="nearest",
        )
        self.model_config = ModelConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            latent_size=self.latent_size,
            gnn_msg_steps=self.gnn_msg_steps,
            hidden_layers=self.hidden_layers,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
        )
        self.task_config = TaskConfig(
            input_variables=self.input_variables,
            target_variables=self.target_variables,
            forcing_variables=self.forcing_variables,
            pressure_levels=levels,
            input_duration=self.input_duration,
        )

        self.all_variables = tuple(set(
            self.input_variables + self.target_variables + self.forcing_variables
        ))

        self.norm = {}
        self.norm["mean"], self.norm["std"], self.norm["stddiff"] = self.load_normalization()


    def subsample_dataset(self, xds, new_time=None):
        """Get the subset of the data that we want in terms of time, vertical levels, and variables

        Args:
            xds (xarray.Dataset): with replay data
            new_time (pandas.Daterange or similar, optional): time vector to select from the dataset

        Returns:
            newds (xarray.Dataset): subsampled/subset that we care about
        """

        # select our vertical levels
        xds = xds.sel(pfull=list(self.pressure_levels), method="nearest")

        # only grab variables we care about
        myvars = list(x for x in self.all_variables if x in xds)
        xds = xds[myvars]

        if new_time is not None:
            xds = xds.sel(time=new_time)

        return xds


    def preprocess(self, xds, batch_index=0, drop_cftime=True):
        """Prepare a single batch for GraphCast

        Args:
            xds (xarray.Dataset): with replay data
            batch_index (int, optional): the index of this batch
            drop_cftime (bool, optional): if True, drop the ``cftime`` and ``ftime`` coordinates that exist in the Replay dataset to avoid future JAX problems (might be helpful to keep them for some debugging cases)

        Returns:
            bds (xarray.Dataset): this batch of data
        """

        # make sure we've subsampled/subset
        bds = self.subsample_dataset(xds)

        bds = bds.rename({
            "pfull": "level",
            "grid_xt": "lon",
            "grid_yt": "lat",
            "time": "datetime",
        })

        # unclear if this is necessary for computation
        bds = bds.sortby("lat", ascending=True)

        bds["time"] = (bds.datetime - bds.datetime[0])
        bds = bds.swap_dims({"datetime": "time"}).reset_coords()
        bds = bds.expand_dims({
            "batch": [batch_index],
        })
        bds = bds.set_coords(["datetime"])

        # cftime is a data_var not a coordinate, but if it's made to be a coordinate
        # it causes crazy JAX problems when making predictions with graphufs.training.run_forward.apply
        # because it thinks something is wrong when the input/output cftime object values are different
        # (even though... of course they will be for prediction)
        # safest to drop here to avoid confusion, along with ftime since it is also not used
        if drop_cftime:
            bds = bds.drop(["cftime", "ftime"])
        return bds


    def get_batches(
        self,
        n_optim_steps=None,
        drop_cftime=True,
        mode="training",
        download_data=True,
    ):
        """Get a dataset with all the batches of data necessary for training

        Note:
            Here we're using target_lead_time as a single value, see graphcast.data_utils.extract ... where it could be multi valued. However, since we are using it to compute the total forecast time per batch soit seems more straightforward as a scalar.

        Args:
            n_optim_steps (int, optional): number of training batches to grab ... number of times we will update the parameters during optimization. If not specified, use as many as are available based on the available training data.
            drop_cftime (bool, optional): may be useful for debugging
            mode (str, optional): can be either "training", "validation" or "testing"
            download_data (bool, optional): download data from GCS
        Returns:
            inputs, targets, forcings (xarray.Dataset): with new dimension "batch"
                and appropriate fields for each dataset, based on the variables in :attr:`task_config`
        """
        # grab the dataset and subsample training portion at desired model time step
        # second epoch onwards should be able to read data locally
        local_data_path = os.path.join(self.local_store_path, "data.zarr")
        if download_data:
            xds = xr.open_zarr(self.data_url, storage_options={"token": "anon"})
        else:
            xds = xr.open_zarr(local_data_path)
        # choose dates based on mode
        if mode == "training":
            start = self.training_dates[ 0]
            end   = self.training_dates[-1]
        elif mode == "testing":
            start = self.testing_dates[ 0]
            end   = self.testing_dates[-1]
        elif mode == "validation":
            start = self.validation_dates[ 0]
            end   = self.validation_dates[-1]
        else:
            raise ValueError("Unknown mode: make sure it is either training/testing/validation")

        # build time vector based on the model, not the data
        delta_t = pd.Timedelta(self.delta_t)
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        all_new_time = pd.date_range(
            start=start,
            end=end,
            freq=delta_t,
            inclusive="both",
        )
        # subsample in time, grab variables and vertical levels we want
        all_xds = self.subsample_dataset(xds, new_time=all_new_time)

        # split dataset into chunks
        chunk_size = len(all_new_time) // self.chunks_per_epoch
        all_new_time_chunks = []
        for i in range(self.chunks_per_epoch):
            if i == self.chunks_per_epoch - 1:
                all_new_time_chunks.append(all_new_time[i * chunk_size:len(all_new_time)])
            else:
                all_new_time_chunks.append(all_new_time[i * chunk_size:(i + 1) * chunk_size])
        print(f"Chunks total: {len(all_new_time_chunks)}")
        for chunk_id, new_time in enumerate(all_new_time_chunks):
            print(f"Chunk {chunk_id}: {new_time[0]} to {new_time[-1]}")

        # iterate over all chunks
        for chunk_id, new_time in enumerate(all_new_time_chunks):

            # chunk start and end times
            start = new_time[0]
            end = new_time[-1]

            # figure out duration of IC(s), forecast, all of training
            input_duration = pd.Timedelta(self.input_duration)
            time_per_forecast = self.target_lead_time + input_duration
            training_duration = end - start

            n_max_forecasts = (training_duration - input_duration) // delta_t
            n_max_optim_steps = n_max_forecasts // self.batch_size
            n_optim_steps = n_max_optim_steps if n_optim_steps is None else n_optim_steps
            n_forecasts = n_optim_steps * self.batch_size

            # note that this max can be violated if we sample with replacement ...
            # but I'd rather just work with epochs and use all the data
            if n_optim_steps > n_max_optim_steps:
                n_optim_steps = n_max_optim_steps
                warnings.warn(f"There's less data than the number of batches requested, reducing n_optim_steps to {n_optim_steps}")

            # create a new time vector with desired delta_t
            # this has to end such that we can pull an entire forecast from the training data
            all_initial_times = pd.date_range(
                start=start,
                end=end - time_per_forecast,
                freq=delta_t,
                inclusive="both",
            )

            # randomly sample without replacement
            # note that GraphCast samples with replacement
            if mode != "testing":
                rstate = np.random.RandomState(seed=self.training_batch_rng_seed)
                forecast_initial_times = rstate.choice(
                    all_initial_times,
                    size=(n_forecasts,),
                    replace=False
                )
            else:
                forecast_initial_times = all_initial_times[:n_forecasts]

            # warnings before we get started
            if pd.Timedelta(self.target_lead_time) > delta_t:
                warnings.warn("ReplayEmulator.get_training_batches: need to rework this to pull targets for all steps at delta_t intervals between initial conditions and target_lead times, at least in part because we need the forcings at each delta_t time step, and the data extraction code only pulls this at each specified target_lead_time")

            # load the dataset in to avoid lots of calls... need to figure out how to do this best

            # subsample in time, grab variables and vertical levels we want
            xds = self.subsample_dataset(all_xds, new_time=new_time)
            if download_data:
                xds.to_zarr(local_data_path, append_dim="time" if os.path.exists(local_data_path) else None)

            xds = xds.load();

            inputs = []
            targets = []
            forcings = []
            inittimes = []
            for i, (k, b) in enumerate(
                itertools.product(range(n_optim_steps), range(self.batch_size))
            ):

                timestamps_in_this_forecast = pd.date_range(
                    start=forecast_initial_times[i],
                    end=forecast_initial_times[i]+time_per_forecast,
                    freq=delta_t,
                    inclusive="both",
                )
                batch = self.preprocess(
                    xds.sel(time=timestamps_in_this_forecast),
                    batch_index=b,
                )

                this_input, this_target, this_forcing = extract_inputs_targets_forcings(
                    batch,
                    target_lead_times=self.target_lead_time,
                    **dataclasses.asdict(self.task_config),
                )

                # fix this later for batch_size != 1
                this_inittimes = batch.datetime.isel(time=0)
                this_inittimes = this_inittimes.to_dataset(name="inittimes")

                # note that the optim_step dim has to be added after the extract_inputs_targets_forcings call
                inputs.append(this_input.expand_dims({"optim_step": [k]}))
                targets.append(this_target.expand_dims({"optim_step": [k]}))
                forcings.append(this_forcing.expand_dims({"optim_step": [k]}))
                inittimes.append(this_inittimes.expand_dims({"optim_step": [k]}))

            inputs = self.combine_chunk(inputs)
            targets = self.combine_chunk(targets)
            forcings = self.combine_chunk(forcings)
            inittimes = self.combine_chunk(inittimes)
            yield inputs, targets, forcings, inittimes


    def load_normalization(self, **kwargs):
        """Load the normalization fields into memory

        Note:
            This uses values for the ``year_progress`` and ``day_progress`` fields in each dataset (mean, std, diffs_std) that were copied from the graphcast demo in order to get moving. These should be recomputed and the lines in this method that set these copied values should be removed.

        Returns:
            mean_by_level, stddev_by_level, diffs_stddev_by_level (xarray.Dataset): with normalization fields
        """

        def open_normalization(component, **kwargs):

            # try to read locally first
            if self.local_store_path is not None:
                local_path = os.path.join(self.local_store_path, os.path.basename(self.norm_urls[component]))

                foundit = False
                if os.path.isdir(local_path):
                    xds = xr.open_zarr(local_path)
                    xds = xds.load()
                    foundit = True
            if not foundit:
                xds = xr.open_zarr(self.norm_urls[component], **kwargs)
                myvars = list(x for x in self.all_variables if x in xds)
                xds = xds[myvars]
                xds = xds.load()
                xds = xds.rename({"pfull": "level"})
                xds.to_zarr(local_path)
            return xds

        mean_by_level = open_normalization("mean")
        stddev_by_level = open_normalization("std")
        diffs_stddev_by_level = open_normalization("stddiff")

        # hacky, just copying these from graphcast demo to get moving
        mean_by_level['year_progress'] = 0.49975101137533784
        mean_by_level['year_progress_sin'] = -0.0019232822626236157
        mean_by_level['year_progress_cos'] = 0.01172127404282719
        mean_by_level['day_progress'] = 0.49861110098039113
        mean_by_level['day_progress_sin'] = -1.0231613285011715e-08
        mean_by_level['day_progress_cos'] = 2.679492657383283e-08

        stddev_by_level['year_progress'] = 0.29067483157079654
        stddev_by_level['year_progress_sin'] = 0.7085840482846367
        stddev_by_level['year_progress_cos'] = 0.7055264413169846
        stddev_by_level['day_progress'] = 0.28867401335991755
        stddev_by_level['day_progress_sin'] = 0.7071067811865475
        stddev_by_level['day_progress_cos'] = 0.7071067888988349

        diffs_stddev_by_level['year_progress'] = 0.024697753562180874
        diffs_stddev_by_level['year_progress_sin'] = 0.0030342521761048467
        diffs_stddev_by_level['year_progress_cos'] = 0.0030474038590028816
        diffs_stddev_by_level['day_progress'] = 0.4330127018922193
        diffs_stddev_by_level['day_progress_sin'] = 0.9999999974440369
        diffs_stddev_by_level['day_progress_cos'] = 1.0

        return mean_by_level, stddev_by_level, diffs_stddev_by_level


    @staticmethod
    def _get_replay_vertical_levels():
        pfull_path = os.path.join(os.path.dirname(__file__), "replay_vertical_levels.yaml")
        with open(pfull_path, "r") as f:
            pfull = yaml.safe_load(f)["pfull"]
        return xr.DataArray(pfull, coords={"pfull": pfull}, dims="pfull")

    def combine_chunk(self, ds_list):
        """Used by the training batch creation code to combine many datasets for optimization"""
        newds = xr.combine_by_coords(ds_list)
        chunksize = {
            "optim_step": 1,
            "batch": -1,
            "time": -1,
            "level": -1,
            "lat": -1,
            "lon": -1,
        }
        chunksize = {k:v for k,v in chunksize.items() if k in newds.dims}
        newds = newds.chunk(chunksize)
        return newds


    def _tree_flatten(self):
        """Pack up everything needed to remake this object.
        Since this class is static, we don't really need anything now, but that will change if we
        set the class attributes with a yaml file.
        In that case the yaml filename will needto be added to the aux_data bit

        See `here <https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree>`_
        for reference.
        """
        children = tuple()
        aux_data = dict() # in the future could be {"config_filename": self.config_filename}
        return (children, aux_data)


    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    ReplayEmulator,
    ReplayEmulator._tree_flatten,
    ReplayEmulator._tree_unflatten,
)
