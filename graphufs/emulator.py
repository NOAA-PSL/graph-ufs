import os
import yaml
import warnings
import itertools
import dataclasses
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
    local_store_path = ""

    # these could be moved to a yaml file later
    # task config options
    input_variables = tuple()
    target_variables = tuple()
    forcing_variables = tuple()
    all_variables = tuple() # this is created in __init__
    pressure_levels = tuple()
    input_duration = None

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

    def __init__(self):

        if self.local_store_path == "":
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


    def preprocess(self, xds, batch_index=0, drop_cftime=True):
        """Prepare a single batch for GraphCast

        Args:
            xds (xarray.Dataset): with replay data
            batch_index (int, optional): the index of this batch
            drop_cftime (bool, optional): if True, drop the ``cftime`` and ``ftime`` coordinates that exist in the Replay dataset to avoid future JAX problems (might be helpful to keep them for some debugging cases)

        Returns:
            bds (xarray.Dataset): this batch of data
        """

        # select our vertical levels
        bds = xds.sel(pfull=list(self.pressure_levels), method="nearest")

        # only grab variables we care about
        myvars = list(x for x in self.all_variables if x in xds)
        bds = bds[myvars]

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


    def get_training_batches(self,
        xds,
        n_optim_steps,
        batch_size,
        delta_t,
        target_lead_time="6h",
        drop_cftime=True,
        ):
        """Get a dataset with all the batches of data necessary for training

        Note:
            Here we're using target_lead_time as a single value, see graphcast.data_utils.extract ... where it could be multi valued. However, since we are using it to compute the total forecast time per batch soit seems more straightforward as a scalar.

        Note:
            Right now there are separate batches initialized from each data point that is separated from the next by ``delta_t``. If ``target_lead_time > delta_t``, then for batch ``n`` and batch ``n+1`` the initial condition for batch ``n+1`` will be between the initial condition and target time of batch ``n`` - i.e., the windows overlap.

        Args:
            xds (xarray.Dataset): the Replay dataset
            n_optim_steps (int): number of training batches to grab ... number of times we will update the parameters during optimization
            batch_size (int): number of samples viewed per mini batch
            delta_t (Timedeltalike): timestep of the desired emulator, e.g. "3h" or "6h". Has to be an integer multiple of the data timestep.
            target_lead_time (str or slice, optional): the lead time to use in the cost function, see graphcast.data_utils.extract_input_target_lead_times

        Returns:
            inputs, targets, forcings (xarray.Dataset): with new dimension "batch"
                and appropriate fields for each dataset, based on the variables in :attr:`task_config`

        Example:

            Create training 100 batches, each batch has a single sample,
            where each sample is made up of error from a 12h forecast,
            where the emulator operates on 6 hour timesteps with a 24h input duration
            (i.e., like graphcast it takes the initial condition and the previous time step to initialize)
            The cftime coordinate is kept here to make things clear.



            >>> gufs = ReplayEmulator()
            >>> xds = #... replay data
            >>> inputs, targets, forcings = gufs.get_training_batches(
                    xds=xds,
                    n_optim_steps=100,
                    batch_size=1,
                    delta_t="6h",
                    target_lead_time="24h",
                    drop_cftime=False,
                )
            >>> for batch in [0, 1]:
            ...:    print("Batch number: ", batch)
            ...:    for lbl, xds in zip(["inputs", "targets", "forcings"], [inputs, targets, forcings]):
            ...:        print(f"{lbl} time (hours): ", xds.sel(batch=batch).time.values / 1e9 / 3600)
            ...:        print(f"{lbl} cftime: ", xds.sel(batch=batch).cftime.values)
            ...:    print("---")
            Batch number:  0
            inputs time (hours):  [-6  0]
            inputs cftime:  [cftime.DatetimeJulian(1993, 12, 31, 18, 0, 0, 0, has_year_zero=False)
             cftime.DatetimeJulian(1994, 1, 1, 0, 0, 0, 0, has_year_zero=False)]
            targets time (hours):  [24]
            targets cftime:  [cftime.DatetimeJulian(1994, 1, 2, 0, 0, 0, 0, has_year_zero=False)]
            forcings time (hours):  [24]
            forcings cftime:  [cftime.DatetimeJulian(1994, 1, 2, 0, 0, 0, 0, has_year_zero=False)]
            ---
            Batch number:  1
            inputs time (hours):  [-6  0]
            inputs cftime:  [cftime.DatetimeJulian(1994, 1, 1, 0, 0, 0, 0, has_year_zero=False)
             cftime.DatetimeJulian(1994, 1, 1, 6, 0, 0, 0, has_year_zero=False)]
            targets time (hours):  [24]
            targets cftime:  [cftime.DatetimeJulian(1994, 1, 2, 6, 0, 0, 0, has_year_zero=False)]
            forcings time (hours):  [24]
            forcings cftime:  [cftime.DatetimeJulian(1994, 1, 2, 6, 0, 0, 0, has_year_zero=False)]

        """

        inputs = []
        targets = []
        forcings = []

        if batch_size > 1:
            warnings.warn("it's not clear how the batch/sample time slices are defined in graphcast or how they are used by optax")

        delta_t = pd.Timedelta(delta_t)
        input_duration = pd.Timedelta(self.input_duration)

        if pd.Timedelta(target_lead_time) > delta_t:
            warnings.warn("need to rework this to pull targets for all steps at delta_t intervals between initial conditions and target_lead times, at least in part because we need the forcings at each delta_t time step, and the data extraction code only pulls this at each specified target_lead_time")

        time_per_sample = target_lead_time + input_duration
        time_per_batch = batch_size * time_per_sample

        # create a new time vector with desired delta_t
        new_time = pd.date_range(
            start=xds["time"].isel(time=0).values,
            end=xds["time"].isel(time=-1).values,
            freq=delta_t,
            inclusive="both",
        )

        # Create a vector of initialization times, currently equal to new_time
        # but see below
        # - freq=delta_t, the ML timestep? if target_lead_time > delta_t, then target windows overlap
        #   ^^ this is what's implemented
        # - freq=target_lead_time?
        # - freq=time_per_batch, so target_lead_time + the input duration (going beyond the extra step(s) needed to initialize)
        batch_initial_times = pd.date_range(
            start=new_time[0],
            end=new_time[-1],
            freq=delta_t,
            inclusive="both",
        )
        if n_optim_steps > len(batch_initial_times)-1:
            n_optim_steps = len(batch_initial_times)-1
            warnings.warn(f"There's less data than the number of batches requested, reducing n_optim_steps to {n_optim_steps}")

        for i, (k, b) in enumerate(
            itertools.product(range(n_optim_steps), range(batch_size))
        ):

            timestamps_in_this_batch = pd.date_range(
                start=batch_initial_times[i],
                end=batch_initial_times[i]+time_per_batch,
                freq=delta_t,
                inclusive="left",
            )
            batch = self.preprocess(
                xds.sel(time=timestamps_in_this_batch),
                batch_index=b,
            )

            this_input, this_target, this_forcing = extract_inputs_targets_forcings(
                batch,
                target_lead_times=target_lead_time,
                **dataclasses.asdict(self.task_config),
            )
            inputs.append(this_input.expand_dims({"optim_step": [k]}))
            targets.append(this_target.expand_dims({"optim_step": [k]}))
            forcings.append(this_forcing.expand_dims({"optim_step": [k]}))

        inputs = xr.merge(inputs)
        targets = xr.merge(targets)
        forcings = xr.merge(forcings)
        return inputs, targets, forcings


    def load_normalization(self, **kwargs):
        """Load the normalization fields into memory

        Note:
            This uses values for the ``year_progress`` and ``day_progress`` fields in each dataset (mean, std, diffs_std) that were copied from the graphcast demo in order to get moving. These should be recomputed and the lines in this method that set these copied values should be removed.

        Returns:
            mean_by_level, stddev_by_level, diffs_stddev_by_level (xarray.Dataset): with normalization fields
        """

        def open_normalization(component, **kwargs):

            # try to read locally first
            local_path = os.path.join(self.local_store_path, os.path.basename(self.norm_urls[component]))
            if os.path.isdir(local_path):
                xds = xr.open_zarr(local_path)
                xds = xds.load()
            else:
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
