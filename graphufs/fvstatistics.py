import logging
from typing import Optional

import numpy as np
import xarray as xr

from graphcast import data_utils

from .fvemulator import fv_vertical_regrid
from .statistics import StatisticsComputer, add_derived_vars

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

    def open_dataset(self, data_vars=None, **tisr_kwargs):
        xds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)

        # subsample in time
        if "time" in xds.dims:
            xds = self.subsample_time(xds)

        xds = add_derived_vars(
            xds,
            transforms=self.transforms,
            compute_tisr=data_utils.TISR in data_vars if data_vars is not None else False,
            **tisr_kwargs,
        )

        # select variables, keeping delz
        if data_vars is not None:
            if isinstance(data_vars, str):
                data_vars = [data_vars]
            if "delz" not in data_vars:
                data_vars.append("delz")

            logging.info(f"{self.name}: computing statistics for {data_vars}")
            xds = xds[data_vars]

        # regrid in the vertical
        logging.info(f"{self.name}: starting vertical regridding")
        xds = fv_vertical_regrid(xds, interfaces=list(self.interfaces))
        return xds
