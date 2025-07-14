# config.py (cleaned up)

from jax import tree_util
import numpy as np
import xarray as xr
from graphufs import FVCoupledEmulator

# === Utility Functions ===
def log(xda):
    cond = xda > 0
    return xr.where(cond, np.log(xda.where(cond)), 0.)

def exp(xda):
    return np.exp(xda)

# === Paths ===
_scratch = "/pscratch/sd/n/nagarwal"

# === Base Trainer Class ===
class BaseOcnTrainer(FVCoupledEmulator):

    # === Data Sources ===
    data_url = {
        "atm": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
               "03h-freq/zarr/fv3.zarr",
        "ocn": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
               "06h-freq/zarr/mom6.zarr",
        "ice": "",
        "land": "",
    }

    norm_urls = {
        "atm": {
            "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                     "06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
            "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                    "06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
            "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                        "06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
        },
        "ocn": {
            "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                     "06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/mean_by_level.zarr",
            "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                    "06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/stddev_by_level.zarr",
            "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/"
                        "06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/diffs_stddev_by_level.zarr",
        },
        "ice": {"mean": "", "std": "", "stddiff": ""},
        "land": {"mean": "", "std": "", "stddiff": ""},
    }

    wb2_obs_url = (
        "gs://weatherbench2/datasets/era5/"
        "1959-2022-6h-64x32_equiangular_conservative.zarr"
    )

    no_cache_data = False

    # === Variable Definitions ===
    # Note: atm_inputs and ocn_inputs must be mutually exclusive
    atm_input_variables = (
        "ugrd10m", "vgrd10m", "spfh2m", "tmp", "spfh",
        "year_progress_sin", "year_progress_cos",
        "day_progress_sin", "day_progress_cos", "land_static",
    )

    ocn_input_variables = (
        "SSH", "LW", "SW", "so", "temp", "uo", "vo", "landsea_mask",
    )

    ice_input_variables = ()
    land_input_variables = ()

    atm_target_variables = ()
    ocn_target_variables = ("SSH", "so", "temp", "uo", "vo")
    ice_target_variables = ()
    land_target_variables = ()

    atm_forcing_variables = (
        "ugrd10m", "vgrd10m", "spfh2m", "tmp", "spfh",
        "year_progress_sin", "year_progress_cos",
        "day_progress_sin", "day_progress_cos",
    )

    ocn_forcing_variables = ("LW", "SW")
    ice_forcing_variables = ()
    land_forcing_variables = ()

    all_variables = ()  # will be defined in __init__

    interfaces = {
        "atm": (950, 1000),
        "ocn": (0, 1, 5, 10, 20, 40, 70, 120, 200, 350, 500),
        "ice": (),
        "land": (),
    }

    # === Transforms ===
    input_transforms = {}
    output_transforms = {}

    # === Time Configuration ===
    delta_t_model = "6h"    # model time step
    delta_t_data = "6h"     # time steps in the data. 
			    # Note: In case of using multiple datasets 
                            # with different time steps, use the most 
			    # coarse time step   
    input_duration = "12h"
    target_lead_time = "6h"

    # the bounds are inclusive
    training_dates = ("1993-12-31T18", "2019-12-31T18")
    validation_dates = ("2022-01-01T00", "2023-10-13T00")
    testing_dates = ("2020-01-01T00", "2021-12-31T18")

    # === Training Configuration ===
    batch_size = 16
    num_batch_splits = 1
    num_epochs = 80
    use_half_precision = False

    # === Model Configuration ===
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    #mesh2grid_edge_normalization_factor = 0.6180338738074472

    # === Loss Configuration ===
    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_channel = True
    weight_loss_per_latitude = True
    weight_loss_per_level = False  # weights both ocean and 
				   # atm vertical levels 
    
    loss_weights_per_variable = {} # weight all of them 
				   # equally

    # === RNG Seeds ===
    # used for initializing the state in gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # === Data Loader Configuration ===
    max_queue_size = 1
    num_workers = 1


# === Register PyTree Node ===
tree_util.register_pytree_node(
    BaseOcnTrainer,
    BaseOcnTrainer._tree_flatten,
    BaseOcnTrainer._tree_unflatten,
)
