import os
import numpy as np
import xarray as xr
import xesmf
import cf_xarray as cfxr

def create_grid_in(
    mom6_grid: xr.Dataset,
) -> xr.Dataset:
    """Convert mom6 grid into a grid that is ready to be used by regridder.
    This comes from here: https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Horizontal_Remapping.html

    Args:
        mom6_grid (xr.Dataset):
            Dataset with all necessary mom6 metadata.

    Returns:
        ds_in_t (xr.Dataset):
            Dataset ready to be used as input grid for tracers.
        ds_in_v (xr.Dataset):
            Dataset ready to be used as input grid for v velocity.
        ds_in_u (xr.Dataset):
            Dataset ready to be used as input grid for u velocity.
    """
    grid_in = xr.Dataset()
    grid_in["lon"] = mom6_grid["geolon"]
    grid_in["lat"] = mom6_grid["geolat"]
    grid_in["lon_u"] = mom6_grid["geolon_u"]
    grid_in["lat_u"] = mom6_grid["geolat_u"]
    grid_in["lon_v"] = mom6_grid["geolon_v"]
    grid_in["lat_v"] = mom6_grid["geolat_v"]
    grid_in['cos_rot'] = mom6_grid["cos_rot"]
    grid_in['sin_rot'] = mom6_grid["sin_rot"]
    ny, nx = grid_in["lon"].shape
    lon_b = np.empty((ny + 1, nx + 1))
    lat_b = np.empty((ny + 1, nx + 1))
    lon_b[1:, 1:] = mom6_grid["geolon_c"].values
    lat_b[1:, 1:] = mom6_grid["geolat_c"].values
    # periodicity
    lon_b[:, 0] = lon_b[:, -1]
    lat_b[:, 0] = lat_b[:, -1]
    # south edge
    dy = (lat_b[2, :] - lat_b[1, :]).mean()
    lat_b[0, 1:] = lat_b[1, 1:] - dy
    lon_b[0, 1:] = lon_b[1, 1:]
    # corner point
    lon_b[0, 0] = lon_b[1, 0]
    lat_b[0, 0] = lat_b[0, 1]
    grid_in["lon_b"] = xr.DataArray(data=lon_b)
    grid_in["lat_b"] = xr.DataArray(data=lat_b)

    # create renamed datasets
    ds_in_t = grid_in[["lon", "lat", "lat_b", "lon_b"]]
    ds_in_u = grid_in[["lon_u", "lat_u", "lat_b", "lon_b"]].rename(
        {"lat_u": "lat", "lon_u": "lon"}
    )
    ds_in_v = grid_in[["lon_v", "lat_v", "lat_b", "lon_b"]].rename(
        {"lat_v": "lat", "lon_v": "lon"}
    )
    ds_rot =  grid_in[['cos_rot','sin_rot']]

    return ds_in_t, ds_in_u, ds_in_v, ds_rot

def create_grid_out(
    lats: np.array,
    lons: np.array,
) -> xr.Dataset:
    """Take lat/lon of our grid and create a grid that is ready for regridder.

    Args:
        lats (np.array):
            Lats of grid_out.
        lons (xr.Dataset):
            Lons of grid_out.

    Returns:
        grid_out (xr.Dataset):
            Grid out that is ready for regridding.
    """
    grid_out = xr.Dataset()
    grid_out["lon"] = xr.DataArray(lons, dims=["lon"])
    grid_out["lat"] = xr.DataArray(lats, dims=["lat"])
    grid_out = grid_out.cf.add_bounds(["lat", "lon"])
    lat_corners = cfxr.bounds_to_vertices(
        bounds=grid_out["lat_bounds"], bounds_dim="bounds", order=None
    )
    lon_corners = cfxr.bounds_to_vertices(
        bounds=grid_out["lon_bounds"], bounds_dim="bounds", order=None
    )
    grid_out = grid_out.assign({"lat_b": lat_corners, "lon_b": lon_corners})
    grid_out = grid_out.drop_vars(["lat_bounds", "lon_bounds"])

    return grid_out


if __name__ == "__main__":

    original_data = xr.load_dataset("/global/cfs/cdirs/m4718/bathymetry/topog_all_edited.nc")[["depth"]]
    grid = xr.load_dataset("mom6_grid_0.25_degree.nc")
    ds_in_t, *_ = create_grid_in(grid)
    rds = xr.open_zarr(
        "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr",
        decode_timedelta=True,
        storage_options={"token": "anon"},
    )
    ds_out = create_grid_out(lons=rds["grid_xt"].values, lats=rds["grid_yt"].values)

    regridder = xesmf.Regridder(
        ds_in=ds_in_t,
        ds_out=ds_out,
        method="conservative",
        periodic=True,
        unmapped_to_nan=True,
    )
    result = regridder(original_data, keep_attrs=True)
    result.to_netcdf("bathymetry.gaussian.0.25-degree.nc")

    subsampled = result.sel(lat=slice(None, None, 4), lon=slice(None, None, 4))
    subsampled.to_netcdf("bathymetry.gaussian.0.25-degree-subsampled.nc")
