from jax import tree_util
import numpy as np
import xarray as xr
import yaml
import os
import sys
from datetime import datetime
from omegaconf import OmegaConf
from graphufs import FVCoupledEmulator
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# === Utility Functions ===
def log(xda):
    cond = xda > 0
    return xr.where(cond, np.log(xda.where(cond)), 0.)

def exp(xda):
    return np.exp(xda)

def load_config(config_path):
    return OmegaConf.load(config_path)
        
# === YAML Config File ===
base_conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_config_ocn_only.yaml")
base_config_path = os.getenv("CONFIG_PATH_OCN_ONLY", base_conf)

# === Trainer Class ===
class OcnTrainer(FVCoupledEmulator):
    def __init__(self, config=None, mpi_rank=None, mpi_size=None):
        base_config = load_config(base_config_path)
        if config:
            if isinstance(config, str):
                custom_config = load_config(config)
            else:
                custom_config = config
            # Merge: custom config overrides base_config
            cfg = OmegaConf.merge(base_config, custom_config)
 
        self.config = OmegaConf.to_container(cfg, resolve=True) 
        
	# === Case ===
        self.case = self.config.get("case", "default")
    
        # === Data Sources ===
        self.data_url = self.config["data_url"]
        self.norm_urls = self.config["norm_urls"]
        self.wb2_obs_url = self.config["wb2_obs_url"]
        self.no_cache_data = self.config["no_cache_data"]
        self.local_store_path = self.config.get("local_store_path")	

        # === Variable Definitions ===
        self.atm_input_variables = tuple(self.config["atm_input_variables"])
        self.ocn_input_variables = tuple(self.config["ocn_input_variables"])
        self.ice_input_variables = tuple(self.config["ice_input_variables"])
        self.land_input_variables = tuple(self.config["land_input_variables"])

        self.atm_target_variables = tuple(self.config["atm_target_variables"])
        self.ocn_target_variables = tuple(self.config["ocn_target_variables"])
        self.ice_target_variables = tuple(self.config["ice_target_variables"])
        self.land_target_variables = tuple(self.config["land_target_variables"])

        self.atm_forcing_variables = tuple(self.config["atm_forcing_variables"])
        self.ocn_forcing_variables = tuple(self.config["ocn_forcing_variables"])
        self.ice_forcing_variables = tuple(self.config["ice_forcing_variables"])
        self.land_forcing_variables = tuple(self.config["land_forcing_variables"])

        #all_variables = ()  # Will be defined in __init__
        self.interfaces = self.config["interfaces"]
  
        # === Time Configuration ===
        self.delta_t_model = self.config["delta_t_model"]
        self.delta_t_data = self.config["delta_t_data"]
        self.input_duration = self.config["input_duration"]
        self.target_lead_time = self.config["target_lead_time"]

        self.training_dates = tuple(self.config["training_dates"])
        self.validation_dates = tuple(self.config["validation_dates"])
        self.testing_dates = tuple(self.config["testing_dates"])

        # === Transforms ===
        self.input_transforms = self.config.get("input_transforms", None)
        self.output_transforms = self.config.get("output_transforms", None)

        # === Training Configuration ===
        self.batch_size = self.config.get("batch_size", 16)
        self.num_batch_splits = self.config.get("num_batch_splits", 1)
        self.num_epochs = self.config.get("num_epochs", 64)
        self.use_half_precision = self.config.get("use_half_precision", False)

        # === Forecast configuration ===
        self.forecast_days = self.config.get("forecast_days", None)
        self.sample_stride = self.config.get("sample_stride", 1)
        self.evaluation_checkpoint_id = self.config.get("evaluation_checkpoint_id", None)
        if self.forecast_days is not None: 
            self.target_lead_time = self.calc_target_lead_time() 

        # === Model Configuration ===
        self.resolution = self.config.get("resolution", 1.0)
        self.mesh_size = self.config.get("mesh_size", 5)
        self.latent_size = self.config.get("latent_size", 512)
        self.lr_peak_value = self.config.get("lr_peak_value", 1e-3)
        self.gnn_msg_steps = self.config.get("gnn_msg_steps", 16)
        self.hidden_layers = self.config.get("hidden_layers", 1)
        self.radius_query_fraction_edge_length = self.config.get("radius_query_fraction_edge_length", 0.6)

	# === Input Gaussian Noise === 
        self.add_gauss_noise = self.config.get("add_gauss_noise", False)
        self.noise_std_as_fraction = self.config.get("noise_std_as_fraction", None)
        self.gauss_noise_seed = self.config.get("gauss_noise_seed", 100)

        # === Loss Configuration ===
        self.weight_loss_per_channel = self.config.get("weight_loss_per_channel", True)
        self.weight_loss_per_latitude = self.config.get("weight_loss_per_latitude", True)
        self.weight_loss_per_level = self.config.get("weight_loss_per_level", False)
        self.loss_weights_per_variable = self.config.get("loss_weights_per_variable", False)

        # === RNG Seeds ===
        self.grad_rng_seed = self.config.get("grad_rng_seed", 0)
        self.init_rng_seed = self.config.get("init_rng_seed", 0)
        self.training_batch_rng_seed = self.config.get("training_batch_rng_seed", 0)

        # === Data Loader Configuration ===
        self.max_queue_size = self.config.get("max_queue_size", 1)
        self.num_workers = self.config.get("num_workers", 1)
	
	# === TensorBoard writer setup === 
        self.log_tensorboard  = self.config.get("log_tensorboard", False)
        self.logdir = self.config.get("logdir", os.path.join(self.local_store_path or "./logs", "tensorboard"))
        if mpi_rank == 0 and self.log_tensorboard:
            self.writer = SummaryWriter(log_dir=self.logdir)
        else:
            self.writer = None

        super().__init__(mpi_rank=mpi_rank, mpi_size=mpi_size)

    def calc_target_lead_time(self):
        # Parse the number of hours from delta_t (assumes format like "6h" or "12h")
        dt = int(self.delta_t_model[:-1])  # in hours

        # total hours of forecasting
        n_autoreg_steps = int(self.forecast_days) * 24

        # List of target lead times: ['6h', '12h', ..., '240h']
        target_lead_time = [f"{n}h" for n in range(dt, n_autoreg_steps + 1, dt)]

        return target_lead_time

    def log_all_metrics(self, dataset: xr.Dataset, epoch: int):
        # Logs all training metrics from a dataset to TensorBoard
        print("dataset recieved for tensorboard logging:", dataset)
        for var_name, arr in dataset.data_vars.items():
            dims = set(arr.dims)
            values = arr.values

            # Skip empty or NaN arrays
            if not np.isfinite(values).any():
                continue

            if dims == {"optim_step"}:
                hist_tag = f"{var_name}/hist"
                scalar_tag = f"{var_name}/scalar"

                global_step = epoch*arr.sizes["optim_step"]

                # Log as a histogram for that epoch
                self.writer.add_histogram(hist_tag, values, epoch)
 
                # Also log as a scalar time series
                for i, v in enumerate(values):
                    self.writer.add_scalar(scalar_tag, v, global_step+i)

            elif dims == {"epoch"}:
                # Log as a scalar for this epoch
                if "_" in var_name:
                    metric, phase = var_name.split("_", 1)
                    scalars = {phase: values} 
                    self.writer.add_scalars(metric, scalars, epoch)
                else: 
                    self.writer.add_scalar(var_name, values, epoch)

            elif dims == {"optim_step", "channel"}:
                values_dict = {}
                for c in arr.channel.values:
                    channel_name = f"channel_{c}"
                    tag = f"{var_name}/{channel_name}"
                    hist_tag = f"{tag}/hist"
                    scalar_tag = f"{tag}/scalar"

                    global_step = epoch*arr.sizes["optim_step"]
                    scalars = arr.sel(channel=c).values
                    
                    # log as a histogram
                    self.writer.add_histogram(hist_tag, scalars, epoch)

                    # log as a scalar time series as well
                    for i, v in enumerate(scalars):
                        self.writer.add_scalar(scalar_tag, v, global_step+i)
                    
            elif dims == {"epoch", "channel"}:
                values_dict = {}
                for c in arr.channel.values:
                    channel_name = f"channel_{c}"
                    tag = f"{var_name}/{channel_name}"
                    val = arr.sel(channel=c).item() 
                    self.writer.add_scalar(tag, val, epoch)
                    
                    values_dict[channel_name] = val
                
                self.writer.add_scalars(var_name, values_dict, epoch)
    
            else:
                # Optionally log histogram if multidimensional
                self.writer.add_histogram(var_name, values, step)

    def finalize(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()

class OcnPreprocessor(OcnTrainer):
    def __init__(self, prototype_id, mpi_rank=None, mpi_size=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ptype_base_config = load_config(f"{script_dir}/{prototype_id}/config.yaml") 
        ptype_preprocessor_config = load_config(f"{script_dir}/{prototype_id}/config_preprocessor.yaml")
        ptype_config = OmegaConf.merge(ptype_base_config, ptype_preprocessor_config)
        super().__init__(config=ptype_config, mpi_rank=mpi_rank, mpi_size=mpi_size)

class OcnPreprocessed(OcnTrainer):
    def __init__(self, prototype_id, mpi_rank=None, mpi_size=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ptype_base_config = load_config(f"{script_dir}/{prototype_id}/config.yaml")
        ptype_preprocessed_config = load_config(f"{script_dir}/{prototype_id}/config_preprocessed.yaml")
        ptype_config = OmegaConf.merge(ptype_base_config, ptype_preprocessed_config)
        super().__init__(config=ptype_config, mpi_rank=mpi_rank, mpi_size=mpi_size)

class OcnEvaluator(OcnTrainer):
    def __init__(self, prototype_id, mpi_rank=None, mpi_size=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ptype_base_config = load_config(f"{script_dir}/{prototype_id}/config.yaml")
        ptype_evaluator_config = load_config(f"{script_dir}/{prototype_id}/config_evaluator.yaml")
        ptype_config = OmegaConf.merge(ptype_base_config, ptype_evaluator_config)
        super().__init__(config=ptype_config, mpi_rank=mpi_rank, mpi_size=mpi_size)
        
# === Register PyTree Node ===
tree_util.register_pytree_node(
    OcnTrainer,
    OcnTrainer._tree_flatten,
    OcnTrainer._tree_unflatten,
)

tree_util.register_pytree_node(
    OcnPreprocessor,
    OcnPreprocessor._tree_flatten,
    OcnPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    OcnPreprocessed,
    OcnPreprocessed._tree_flatten,
    OcnPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    OcnEvaluator,
    OcnEvaluator._tree_flatten,
    OcnEvaluator._tree_unflatten
)
