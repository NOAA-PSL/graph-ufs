import logging
from typing import Optional
from copy import copy

import numpy as np
import xarray as xr

from graphcast import data_utils

from .diagnostics import prepare_diagnostic_functions
from .fvemulator import fv_vertical_regrid
from .statistics import StatisticsComputer, add_derived_vars, add_transformed_vars

class FVStatisticsComputer(StatisticsComputer):
    """Class for computing normalization statistics, using a delz vertical weighted average
    in the computation.

    For other attributes and docs, see :class:`StatisticsComputer`

    Note:
        The operations in open_dataset should align with FVEmulator.subsample_dataset.
        For now, this means we are computing transformations (done in add_derived_vars),
        then doing vertical regridding.

    Attributes:
        interfaces (array_like): with approximate values of vertical grid interfaces to
            grab from the parent dataset, using nearest neighbor (i.e., passing 100 will give
            return the closest value of 101.963245 or whatever it is or whatever it is)
        """


    def __init__(
        self,
        path_in: str,
        path_out: str,
        interfaces: tuple | list | np.ndarray,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        time_skip: Optional[int] = None,
        open_zarr_kwargs: Optional[dict] = None,
        to_zarr_kwargs: Optional[dict] = None,
        load_full_dataset: Optional[bool] = False,
        transforms: Optional[dict] = None,
    ):
        super().__init__(
            path_in=path_in,
            path_out=path_out,
            start_date=start_date,
            end_date=end_date,
            time_skip=time_skip,
            open_zarr_kwargs=open_zarr_kwargs,
            to_zarr_kwargs=to_zarr_kwargs,
            load_full_dataset=load_full_dataset,
            transforms=transforms,
        )
        self.interfaces = interfaces

    def open_dataset(self, data_vars=None, diagnostics=None, **tisr_kwargs):
        xds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)

        # subsample in time
        if "time" in xds.dims:
            xds = self.subsample_time(xds)

        logging.info(f"{self.name}: Adding any derived variables")
        xds = add_derived_vars(
            xds,
            compute_tisr=data_utils.TISR in data_vars if data_vars is not None else False,
            **tisr_kwargs,
        )

        if diagnostics is not None:
            if isinstance(diagnostics, str):
                diagnostics = [diagnostics]
            diagnostic_mappings = prepare_diagnostic_functions(diagnostics)

        # select variables, keeping delz
        # note that all of this complicated logic is because we want to regrid before computing
        # transformations and diagnostics, but we don't want to create the regridding task graph
        # for ALL the variables in the dataset (which would be the case if we simply grabbed the variables
        # after transformations/diagnostics), because that could be HUUUUGE
        if data_vars is not None:
            local_data_vars = copy(data_vars)
            if isinstance(data_vars, str):
                data_vars = [data_vars]
                local_data_vars = [local_data_vars]

            # if the transformed variables are desired, need to hang onto
            # the original, not transformed variable, since we vertically average then transform
            for key, mapping in self.transforms.items():
                transformed_key = f"{mapping.__name__}_{key}"
                do_transformed_var = any(transformed_key == dv for dv in local_data_vars)
                original_var_not_in_list = all(key != dv for dv in local_data_vars)

                if do_transformed_var:
                    local_data_vars.remove(transformed_key)
                if do_transformed_var and original_var_not_in_list:
                    local_data_vars.append(key)

            # now if we want diagnostics, make sure the required variables are there
            if diagnostics is not None:
                for key, required_variables in diagnostic_mappings["required_variables"].items():
                    do_this_diagnostic = any(key == dv for dv in diagnostics)
                    if do_this_diagnostic:
                        for required_var in required_variables:
                            if required_var not in local_data_vars:
                                local_data_vars.append(required_var)

            if "pfull" in xds[local_data_vars].dims:
                xds = xds[local_data_vars+["delz"]]
            else:
                xds = xds[local_data_vars]

        # regrid in the vertical
        if "pfull" in xds.dims:
            logging.info(f"{self.name}: starting vertical regridding")
            xds = fv_vertical_regrid(xds, interfaces=list(self.interfaces))

        logging.info(f"{self.name}: Adding any transformed variables")
        xds = add_transformed_vars(
            xds,
            transforms=self.transforms,
        )

        if diagnostics is not None:
            logging.info(f"{self.name}: computing diagnostics {diagnostics}")
            for key, func in diagnostic_mappings["functions"].items():
                xds[key] = func(xds)


        if data_vars is not None:
            selvars = data_vars + list(diagnostics) if diagnostics is not None else data_vars
            for key in ["phalf", "ak", "bk"]:
                if key in xds:
                    selvars.append(key)
            xds = xds[selvars]

        return xds
