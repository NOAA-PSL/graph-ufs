import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cmocean

import graphufs
from graphufs.spatialmap import SpatialMap
from graphufs.utils import open_zarr

_scratch = "/pscratch/sd/t/timothys/p2p/uvwc"
_date = "2022-12-11T09"
_fhr = 24

if __name__ == "__main__":
    plt.style.use("graphufs.plotstyle")

    gds = open_zarr(f"{_scratch}/inference/validation/graphufs.240h.zarr")
    gds = gds.sel(time=_date, fhr=_fhr)

    era = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        storage_options={"token": "anon"},
    ).sel(
        time=[pd.Timestamp(_date) + pd.Timedelta(hours=_fhr)],
        level=[500],
    )

    gds["10m_wind_speed"] = np.sqrt(gds["ugrd10m"]**2 + gds["vgrd10m"]**2)
    era["10m_wind_speed"] = np.sqrt(era["10m_u_component_of_wind"]**2 + era["10m_v_component_of_wind"]**2)

    # now the plotting
    mapper = SpatialMap()

    for gfld, efld in zip(
        ["tmp2m", "10m_wind_speed"],
        ["2m_temperature", "10m_wind_speed"],
    ):
        fig, axs = mapper.plot(gds[gfld], era[efld])
        fig.savefig(
            f"{_scratch}/figures/graphufs_and_era5_{gfld}_24h.jpeg",
            bbox_inches="tight",
            dpi=300,
        )
