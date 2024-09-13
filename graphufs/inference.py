import logging
import xarray as xr
import dask

def swap_batch_time_dims(xds, inittimes):

    xds = xds.rename({"time": "lead_time"})

    # create "time" dimension = t0
    xds["time"] = xr.DataArray(
        inittimes,
        coords=xds["batch"].coords,
        dims=xds["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    # swap logical batch for t0
    xds = xds.swap_dims({"batch": "time"}).drop_vars("batch")
    return xds


def store_container(path, xds, time, **kwargs):

    if "time" in xds:
        xds = xds.isel(time=0, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = ("time",) + xds[key].dims
        coords = {"time": time, **dict(xds[key].coords)}
        shape = (len(time),) + xds[key].shape
        chunks = (1,) + tuple(-1 for _ in xds[key].dims)

        container[key] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=chunks,
                dtype=xds[key].dtype,
            ),
            coords=coords,
            dims=dims,
            attrs=xds[key].attrs.copy(),
        )
    container.to_zarr(path, compute=False, **kwargs)
    logging.info(f"Stored container at {path}")
