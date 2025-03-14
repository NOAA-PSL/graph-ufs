import logging
import warnings
import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
import yaml
try:
    import flox
    _has_flox = True
except ImportError:
    _has_flox = False
from typing import Optional
from ufs2arco import Layers2Pressure
from graphcast.graphcast import ModelConfig, TaskConfig
from graphcast import data_utils
from .coupledemulator import ReplayCoupledEmulator

class FVCoupledEmulator(ReplayCoupledEmulator):
    interfaces = None # Note the these values can be approximate, we'll grab nearest neighbors to Replay dataset


    def __init__(self, mpi_rank=None, mpi_size=None):

        if not _has_flox:
            warnings.warn("Could not import flox, install with 'conda install -c conda-forge flox' for faster volume averaging (i.e. groupby operations)")

        if self.local_store_path is None:
            warnings.warn("FVCoupledEmulator.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")
       
        # Combine input and target variables from all components                                                      
        self.input_variables = tuple(set(self.atm_input_variables+self.ocn_input_variables+                           
            self.ice_input_variables+self.land_input_variables))                                                      
        self.target_variables = tuple(set(self.atm_target_variables+self.ocn_target_variables+                        
            self.ice_target_variables+self.land_target_variables))                                                    
        self.forcing_variables = tuple(set(self.atm_forcing_variables+self.ocn_forcing_variables+                     
            self.ice_forcing_variables+self.land_forcing_variables)) 

        if any(x not in self.input_variables for x in self.target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict target variables that are not also inputs")

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

        latitude, longitude = self._get_replay_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)

        # finite-volume vertical regridding
        nds_atm = get_new_vertical_grid(list(self.interfaces["atm"]), "atm")
        self.atm_levels = list(nds_atm["pfull"].values)
        #self.pressure_levels = tuple(nds_atm["pfull"].values)

        if self.interfaces["ocn"]:
            nds_ocn = get_new_vertical_grid(list(self.interfaces["ocn"]), "ocn")
            self.ocn_levels = list(nds_ocn["z_l"].values)

        self.model_config = ModelConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            latent_size=self.latent_size,
            gnn_msg_steps=self.gnn_msg_steps,
            hidden_layers=self.hidden_layers,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
        )
        # try/except logic to support original graphcast.graphcast.TaskConfig
        # since I couldn't get inspect.getfullargspec to work
        try:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(set(self.atm_levels)),
                ocn_vert_levels=sorted(tuple(set(self.ocn_levels))),
                input_duration=self.input_duration,
                longitude=self.longitude,
                latitude=self.latitude,
            )
        except ValueError:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(set(self.atm_levels)),
                ocn_vert_levels=sorted(tuple(set(self.ocn_levels))),
                input_duration=self.input_duration,
            )


        self.all_variables = tuple(set(
            self.input_variables + self.target_variables + self.forcing_variables
        ))

        # convert some types
        self.delta_t = pd.Timedelta(self.delta_t)
        self.input_duration = pd.Timedelta(self.input_duration)
        lead_times, duration = data_utils._process_target_lead_times_and_get_duration(self.target_lead_time)
        self.forecast_duration = duration

        logging.debug(f"target_lead_time: {self.target_lead_time}")
        logging.debug(f"lead_times: {lead_times}")
        logging.debug(f"self.forecast_duration: {self.forecast_duration}")
        logging.debug(f"self.time_per_forecast: {self.time_per_forecast}")
        logging.debug(f"self.n_input: {self.n_input}")
        logging.debug(f"self.n_forecast: {self.n_forecast}")
        logging.debug(f"self.n_target: {self.n_target}")

        # set normalization here so that we can jit compile with this class
        # a bit annoying, have to copy datatypes here to avoid the Ghost Bus problem
        self.norm_urls = self.norm_urls.copy()
        self.norm = dict()
        self.stacked_norm = dict()
        self.set_normalization()
        self.set_stacked_normalization()

        # TOA Incident Solar Radiation integration period
        if self.tisr_integration_period is None:
            self.tisr_integration_period = self.delta_t

    def subsample_dataset(self, xds, es_comp="atm", new_time=None):
    
        if es_comp=="atm":
            myvars = list(x for x in list(set(self.atm_input_variables+self.atm_target_variables+self.atm_forcing_variables)) if x in xds)
            # make sure that we have 'delz' for the vertical averaging
            if "delz" not in myvars:
                myvars.append("delz")
                
            xds = xds[myvars]
            if self.interfaces["atm"]:
                xds = fv_vertical_regrid_atm(xds, interfaces=self.interfaces["atm"])
            
            # if we didn't want delz and just kept it for regridding, remove it here
            xds = xds[[x for x in self.all_variables if x in xds]]
            
        elif es_comp=="ocn":
            myvars = list(x for x in list(set(self.ocn_input_variables+self.ocn_target_variables+self.ocn_forcing_variables)) if x in xds)
            xds = xds[myvars]
            if self.interfaces["ocn"]:
                xds = fv_vertical_regrid_ocn(xds, interfaces=self.interfaces["ocn"], keep_dz=False)
                # if landsea_mask is specified, this would also get FV regridded in the above step, which is not entirely correct. We must
                # overwrite this to a landsea_mask diagnosed from one of the 3D FV regridded variable.
                xds, mask = diagnose_and_append_ocean_mask(xds)
                # if the correct statistics for mask is not present, recompute it and append
                #for moment in ["mean", "std"]:
                #        self.append_diagnosed_mask_statistics(moment, mask, diag_mask_var, "ocn") 
        
        elif es_comp=="ice":
            myvars = list(x for x in list(set(self.ice_input_variables+self.ice_target_variables+self.ice_forcing_variables)) if x in xds)
            xds = xds[myvars]
            if self.interfaces["ice"]:
                xds = fv_vertical_regrid_ice(xds, interfaces=self.interfaces["ice"])
        
        elif es_comp=="land":
            myvars = list(x for x in list(set(self.land_input_variables+self.land_target_variables+self.land_forcing_variables)) if x in xds)
            xds = xds[myvars]
            if self.interfaces["land"]:
                xds = fv_vertical_regrid_land(xds, interfaces=self.interfaces["land"])
                    
        if new_time is not None and "time" in xds.dims:
            xds = xds.sel(time=new_time)
        
        # perform transform after vertical averaging, less subsceptible to noisy results
        if es_comp == "atm":
            xds = self.transform_variables(xds)
        
        xds = xds.fillna(0)
        return xds

    def append_diagnosed_mask_statistics(self, moment, ds_mask, diag_mask_var, es_comp="ocn"):
        local_path = os.path.join(
            self.local_store_path,
            "normalization",
            os.path.basename(self.norm_urls[es_comp][moment]),
        )
        print("appending statistics of the diagnosed ocean mask")
        if os.path.isdir(local_path):
            xds = xr.open_zarr(local_path,)
            if diag_mask_var not in xds:
                xds = xds.load()
                # add mean as 0
                if moment.lower() == "mean":
                    xds[diag_mask_var] = xr.DataArray(
                            np.zeros(ds_mask.sizes["z_l"]),
                            dims=["z_l"],
                            coords={"z_l":ds_mask.coords["z_l"]},
                            attrs={"description": "min over lat, lon for max-min normalization: replace mean in \
                                    standardization by this value" },
                            )
                # add std as 1
                elif moment.lower() == "std":
                    xds[diag_mask_var] = xr.DataArray(
                            np.ones(ds_mask.sizes["z_l"]),
                            dims=["z_l"],
                            coords={"z_l":ds_mask.coords["z_l"]},
                            attrs={"description": "max-min over lat, lon for max-min normalization: replace std in \
                                    standardization by this value" },
                            )

            xds.close()
            # delete any existing stacked norm statistic and re-stack
            stacked_norm_path = os.path.join(
                self.local_store_path,
                "stacked-normalization",
            )
            if os.path.exists(stacked_norm_path):
                try:
                    shutil.rmtree(stacked_norm_path)
                except OSError as e:
                    print(f"Error deleting {stacked_norm_path}: {e}")
            else:
                print(f"Folder {stacked_norm_path} does not exist")
            
            self.set_stacked_normalization()

        else:
            warnings.warn(f"local statistics store {local_path} not found. no statistics stored for diagnosed mask")
        return
        
def get_new_vertical_grid(interfaces, comp):

    # Create the parent vertical grid via layers2pressure object
    if comp.lower()=="atm":
        replay_layers = Layers2Pressure()
        phalf = replay_layers.phalf.sel(phalf=interfaces, method="nearest")

        # Make a new Layers2Pressure object, which has the subsampled vertical grid
        # note that pfull gets defined internally
        child_layers = Layers2Pressure(
                ak=replay_layers.xds["ak"].sel(phalf=phalf),
                bk=replay_layers.xds["bk"].sel(phalf=phalf),
        )
        nds = child_layers.xds.copy(deep=True)
    
    elif comp.lower()=="ocn":
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)
        replay_mom6_interface = xr.DataArray(np.array(vcoords["z_i"]),
                coords={"z_i":np.array(vcoords["z_i"])},
                dims=["z_i"],
                )
        nz_i = replay_mom6_interface.sel(z_i=interfaces, method='nearest').values
        nz_l = (nz_i[1:] + nz_i[:-1])/2
        nds = _get_ocn_xds(nz_i, nz_l)
        
    return nds

def _get_ocn_xds(
        z_i: np.ndarray,
        z_l: Optional[np.ndarray] = None,
    ) -> xr.Dataset:
    
        if z_i is not None:
            z_i = xr.DataArray(
                z_i,
                coords={"z_i":z_i},
                dims=["z_i"],
            )
            if z_l is not None:
                z_l = xr.DataArray(
                    z_l,
                    coords={"z_l":z_l},
                    dims=["z_l"],
                )

                xds = xr.Dataset({
                    "z_i": z_i,
                    "z_l": z_l,
                })
                xds = xds.set_coords(["z_i", "z_l"])
            
            else:
                xds = xr.Dataset({
                    "z_i":z_i,
                })
                xds = xds.set_coords(["z_i"])
            
            return xds
        
        else:
            raise ValueError("Both z_i and z_l are empty")
            

def fv_vertical_regrid_ocn(xds, interfaces, keep_dz=False):
    """Vertically regrid an ocean dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces
    
    Args:
        xds (xr.Dataset)
        interfaces (tuple like interface locations)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """
    
    vars2d = [x for x in xds.data_vars if not "z_l" in xds[x].dims]
    vars3d = [x for x in xds.data_vars if "z_l" in xds[x].dims]
    
    if vars3d:
        logging.info(f"3D ocean variable detected:{vars3d} ")
        nds = get_new_vertical_grid(list(interfaces), "ocn")
        # Regrid static layer thickness and get weighting
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)
        xa_interface = xr.DataArray(np.array(vcoords["z_i"]),
                coords={"z_i":np.array(vcoords["z_i"])},
                dims=["z_i"],
        )

        xa_dz = xa_interface.diff(dim="z_i")
        xa_dz = xa_dz.assign_coords({"z_i":xds.coords["z_l"].values})
        xa_dz = xa_dz.rename({"z_i":"z_l"})
        nds["dz"] = xa_dz.groupby_bins(
                "z_l",
                bins=nds["z_i"],
        ).sum()
        dz_inverse = 1/nds["dz"]
            
        # do the regridding for all variables now.
        for key in vars3d:
            with xr.set_options(keep_attrs=True):
                weighted_mult = xds[key]*xa_dz
                nds[key] = dz_inverse*(
                    weighted_mult.groupby_bins(
                        "z_l",
                        bins=nds["z_i"],
                    ).sum()
                )
            nds[key].attrs = xds[key].attrs.copy()
            # set the coordinates
            nds = nds.set_coords("z_l")
            nds["z_l_bins"] = nds["z_l_bins"].swap_dims({"z_l_bins": "z_l"})
            with xr.set_options(keep_attrs=True):
                nds[key] = nds[key].swap_dims({"z_l_bins": "z_l"})
            nds[key].attrs["regridding"] = f"layer thickness weighted average in vertical, new coordinate bounds represented by 'z_l_bins'"

            nds = nds.drop_vars("z_l_bins")
        nds["dz"] = nds["dz"].swap_dims({"z_l_bins": "z_l"})
        if not keep_dz:
            nds = nds.drop_vars("dz")
        
    if vars2d:
        nds2d = no_fvregrid_to_2d(xds, vars2d)
        nds = xr.merge([nds, nds2d])

    return nds


def fv_vertical_regrid_atm(xds, interfaces, keep_delz=True):
    """Vertically regrid an atm dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (tuple like interface locations)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """

    vars2d = [x for x in xds.data_vars if not "pfull" in xds[x].dims]
    vars3d = [x for x in xds.data_vars if "pfull" in xds[x].dims]
    
    if vars3d:
        logging.info(f"3D variable detected:{vars3d} ")
        # create a new dataset with the new vertical grid 
        nds = get_new_vertical_grid(list(interfaces), "atm")

        # if the dataset has somehow already renamed pfull -> level, rename to pfull for Layers2Pressure computations
        has_level_not_pfull = False 
        if "level" in xds.dims and "pfull" not in xds.dims:
            with xr.set_options(keep_attrs=True):
                xds = xds.rename({"level": "pfull"}) 

        # Regrid vertical distance, and get weighting
        nds["delz"] = xds["delz"].groupby_bins(
            "pfull",
            bins=nds["phalf"],
        ).sum()
        new_delz_inverse = 1/nds["delz"]
            
        for key in vars3d:
            with xr.set_options(keep_attrs=True):
                nds[key] = new_delz_inverse * (
                    (
                        xds[key]*xds["delz"]
                    ).groupby_bins(
                        "pfull",
                        bins=nds["phalf"],
                    ).sum()
                )
            nds[key].attrs = xds[key].attrs.copy()
            
            # set the coordinates
            nds = nds.set_coords("pfull")
            nds["pfull_bins"] = nds["pfull_bins"].swap_dims({"pfull_bins": "pfull"})
            with xr.set_options(keep_attrs=True):
                nds[key] = nds[key].swap_dims({"pfull_bins": "pfull"})

            nds[key].attrs["regridding"] = "delz weighted average in vertical,new coordinate bounds represented by 'pfull_bins'"
        # unfortunately, cannot store the pfull_bins due to this issue: https://github.com/pydata/xarray/issues/2847  
        nds = nds.drop_vars("pfull_bins")

    if not keep_delz:
        nds = nds.drop_vars("delz")

    if vars2d:
        nds2d = no_fvregrid_to_2d(xds, vars2d)
        nds = xr.merge([nds, nds2d])
    
    return nds


def fv_vertical_regrid_ice(xds, interfaces):
    """Vertically regrid an ice dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (tuple like interface locations)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """

    if interfaces:
        raise NotImplementedError("Error: non-zero vertical ice interfaces found. 3D ice variable are not supported")
    else:
        vars2d = [x for x in xds.data_vars]
        if vars2d:
            nds = no_fvregrid_to_2d(xds, vars2d)
    
    return nds

def fv_vertical_regrid_land(xds, interfaces):
    """Vertically regrid a land dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (tuple like interface locations)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """

    if interfaces:
        raise NotImplementedError("Error: non-zero vertical land interfaces found. 3D land variable are not supported")
    else:
        vars2d = [x for x in xds.data_vars]
        if vars2d:
            nds = no_fvregrid_to_2d(xds, vars2d)
    
    return nds


def no_fvregrid_to_2d(xds, varlist):
    """No FV vertical regridding is applied to 2D variables
    
    Args:
        xds (xr.Dataset)
        varlist (list of 2d variables)
    
    Returns:
        nds (xr.Dataset): a copy of the 2D variables
    """
    logging.info("2D variable detected: no regridding applied")
        
    nds = xr.Dataset(
            data_vars={}, 
            attrs=xds.attrs,
    )
    for key in varlist:
        nds[key] = xds[key]
    
    return nds

def diagnose_and_append_ocean_mask(xds):
    """
    Diagnose and append a 3D ocean mask using a FV regridded variable. Here, the mask
    is diagnosed using a 3D salinity snapshot. We use salinity because fvregridding is 
    removing all nan values as a result of the .sum() operation. We cannot even use 
    skipna=False there because that may end up considering an ocean column as land column 
    if only the last of the grid box in that column is land (say a topographic tip). Salinity
    can be used to bypass this problem as a zero salinity is not within its range and therefore
    it must be a land point.
    
    Args:
    xds: FV regridded ocean dataset without the land-sea mask.

    Returns:
    xds: ocean dataset with the diagnosed landsea_mask
    """
    # pick a snapshot
    try:
        quantity = xds.so.isel(time=0)
    except:
        raise KeyError("salinity is required for computing landsea mask")
    
    # create the mask and add attributes
    ocean_mask = xr.where(quantity==0, 0, 1).astype(xds.so.dtype)
    ocean_mask = ocean_mask.assign_attrs(long_name="land-sea mask (land=0, sea/ice=1)", units="None",)
    
    # deleting any time dimensions
    not_requires_dims = ["cftime", "ftime", "time"]
    for key in not_requires_dims:
        if key in ocean_mask.dims:
            ocean_mask = ocean_mask.reset_coords(names=key, drop=True)
    
    # append landsea_mask to xds 
    if "landsea_mask" in xds:
        xds["landsea_mask"] = ocean_mask
        logging.info("updated landsea_mask variable with the diagnosed one")
    else:
        xds = xds.assign(landsea_mask=ocean_mask)
        logging.info("appended new diagnosed 3d ocean mask")

    return xds, ocean_mask


def fv_vertical_regrid(xds, interfaces, keep_delz=True, keep_dz=True):
    """Vertically regrid a dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in 
    interfaces. Note: here the input dataset may contain any possible combination 
    of variables from ocn, atmosphere, ice, and land components, unlike other 
    earth system component specific functions defined above. This is useful for 
    generating fvstatistics for the coupled configuration. 
    
    Args:
        xds (xr.Dataset)
        interfaces (interfaces for the given component. Note that it can only be tuple or list or dictionary)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """
    # 2D or 3D: no vertical regridding required for 2D
    vars2d = [x for x in xds.data_vars if not ("pfull" in xds[x].dims or "z_l" in xds[x].dims)]
    vars3d = {}
    vars3d["atm"] = [x for x in xds.data_vars if "pfull" in xds[x].dims]
    vars3d["ocn"] = [x for x in xds.data_vars if "z_l" in xds[x].dims]
    has_3Datm = False
    has_3Docn = False
    has_2D = False

    if vars3d:
        if vars3d["atm"]:
            has_3Datm = True
            logging.info(f"3D atmospheric variable detected:{vars3d} ")
            
            # create a new dataset with the new vertical grid 
            if isinstance(interfaces, dict):
            #    raise ValueError("interfaces cannot be a dictionary. Slice and pass interfaces as a tuple/list for the atm component")
                nds3Datm = get_new_vertical_grid(list(interfaces["atm"]), "atm")
            else:
                nds3Datm = get_new_vertical_grid(list(interfaces), "atm")
            # if the dataset has somehow already renamed pfull -> level, rename to pfull for Layers2Pressure computations
            has_level_not_pfull = False 
            if "level" in xds.dims and "pfull" not in xds.dims:
                with xr.set_options(keep_attrs=True):
                    xds = xds.rename({"level": "pfull"}) 

            # Regrid vertical distance, and get weighting
            nds3Datm["delz"] = xds["delz"].groupby_bins(
                "pfull",
                bins=nds3Datm["phalf"],
            ).sum()
            new_delz_inverse = 1/nds3Datm["delz"]

            for key in vars3d["atm"]:
                # Regrid the variable
                with xr.set_options(keep_attrs=True):
                    nds3Datm[key] = new_delz_inverse * (
                        (
                            xds[key]*xds["delz"]
                        ).groupby_bins(
                            "pfull",
                            bins=nds3Datm["phalf"],
                        ).sum()
                    )
                nds3Datm[key].attrs = xds[key].attrs.copy()
            
                # set the coordinates
                nds3Datm = nds3Datm.set_coords("pfull")
                nds3Datm["pfull_bins"] = nds3Datm["pfull_bins"].swap_dims({"pfull_bins": "pfull"})
                with xr.set_options(keep_attrs=True):
                    nds3Datm[key] = nds3Datm[key].swap_dims({"pfull_bins": "pfull"})

                nds3Datm[key].attrs["regridding"] = "delz weighted average in vertical,new coordinate bounds represented by 'pfull_bins'"
                # unfortunately, cannot store the pfull_bins due to this issue: https://github.com/pydata/xarray/issues/2847  
                nds3Datm = nds3Datm.drop_vars("pfull_bins")

            if not keep_delz:
                nds3Datm = nds3Datm.drop_vars("delz")
            
        if vars3d["ocn"]:
            has_3Docn = True
            logging.info(f"3D ocean variable detected:{vars3d} ")
            # create a new dataset with the new vertical grid
            if isinstance(interfaces, dict):
            #    raise ValueError("interfaces cannot be a dictionary. Slice and pass interfaces as a tuple/list for the ocn component")
                nds3Docn = get_new_vertical_grid(list(interfaces["ocn"]), "ocn")
            else:
                nds3Docn = get_new_vertical_grid(list(interfaces), "ocn")
            # Regrid static layer thickness and get weighting
            vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"replay_vertical_levels.yaml")
            with open(vcoord_path, "r") as f:
                vcoords = yaml.safe_load(f)
            xa_interface = xr.DataArray(np.array(vcoords["z_i"]),
                    coords={"z_i":np.array(vcoords["z_i"])},
                    dims=["z_i"],
                    )

            xa_dz = xa_interface.diff(dim="z_i")
            xa_dz = xa_dz.assign_coords({"z_i":xds.coords["z_l"].values})
            xa_dz = xa_dz.rename({"z_i":"z_l"})
            nds3Docn["dz"] = xa_dz.groupby_bins(
                    "z_l",
                    bins=nds3Docn["z_i"],
            ).sum()
            dz_inverse = 1/nds3Docn["dz"]
            # do the regridding for all variables now.
            for key in vars3d["ocn"]:
                with xr.set_options(keep_attrs=True):
                    weighted_mult = xds[key]*xa_dz
                    nds3Docn[key] = dz_inverse*(
                        weighted_mult.groupby_bins(
                            "z_l",
                            bins=nds3Docn["z_i"],
                        ).sum()
                    )
                nds3Docn[key].attrs = xds[key].attrs.copy()
                # set the coordinates
                nds3Docn = nds3Docn.set_coords("z_l")
                nds3Docn["z_l_bins"] = nds3Docn["z_l_bins"].swap_dims({"z_l_bins": "z_l"})
                with xr.set_options(keep_attrs=True):
                    nds3Docn[key] = nds3Docn[key].swap_dims({"z_l_bins": "z_l"})
                nds3Docn[key].attrs["regridding"] = "layer thickness weighted average in vertical, new coordinate bounds represented by 'z_l_bins'"

                nds3Docn = nds3Docn.drop_vars("z_l_bins")
            nds3Docn["dz"] = nds3Docn["dz"].swap_dims({"z_l_bins": "z_l"})
            if not keep_dz:
                nds3Docn = nds3Docn.drop_vars("dz")
            
    if vars2d:
        has_2D = True
        logging.info("2D variable detected: no regridding applied")
        nds2D = no_fvregrid_to_2d(xds, vars2d)
        

    if has_3Datm:
        if has_3Docn:
            if has_2D:
                nds = xr.merge([nds3Datm, nds3Docn, nds2D])
            else:
                nds = xr.merge([nds3Datm, nds3Docn])
        else:
            if has_2D:
                nds = xr.merge([nds3Datm, nds2D])
            else:
                nds = nds3Datm.copy()
    elif has_3Docn:
        if has_2D:
            nds = xr.merge([nds3Docn, nds2D])
        else:
            nds = nds3Docn.copy()
    else:
        nds = nds2D.copy()

    return nds
