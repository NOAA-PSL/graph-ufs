import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

class SpatialMap():
    defaults = {
        "tmp2m": {
            "vmax": 30,
            "vmin": -10,
            "cmap": "cmo.thermal",
            "label": "2m Temperature ($^\circ$C)",
        },
        "10m_wind_speed": {
            "vmax": 25,
            "vmin": 0,
            "cmap": "cmo.speed",
            "label": "10m Horizontal Wind Speed (m/s)",
        },
    }

    central_longitude = -80
    central_latitude = 20

    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            try:
                getattr(self, key)
                setattr(self, key, val)
            except:
                print(f"No attr {key}, ignoring kwarg")

    @staticmethod
    def get_figsize(ncols):
        return (5*ncols, 6)


    def plot(self, gda, tda, subselect=False, **kwargs):

        date = tda.time.values[0]

        if subselect:
            tda = tda.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))

        fig, axs = plt.subplots(
            1,2,
            figsize=self.get_figsize(2),
            constrained_layout=True,
            subplot_kw={
                "projection": ccrs.Orthographic(
                    central_longitude=self.central_longitude,
                    central_latitude=self.central_latitude,
                ),
            },
        )
        for key in ["vmax", "vmin", "cmap"]:
            if key not in kwargs:
                kwargs[key] = self.defaults[gda.name][key] if gda.name in self.defaults.keys() else None


        if gda.name in self.defaults.keys():
            label = self.defaults[gda.name]["label"]
        else:
            label = f"{gda.name} ({tda.attrs['units']})"

        if gda.name == "tmp2m":
            gda = gda - 273.15
            tda = tda - 273.15

        extend, kwargs["vmin"], kwargs["vmax"] = get_extend([gda, tda], kwargs["vmin"], kwargs["vmax"])
        kw = {"transform": ccrs.PlateCarree(), "add_colorbar": False, **kwargs}

        p = gda.plot(ax=axs[0], **kw)
        tda.plot(ax=axs[1], x="longitude", **kw)

        [ax.add_feature(cf.COASTLINE) for ax in axs];
        if "fhr" in gda.coords:
            axs[0].set(title=f"GraphUFS, {str(gda.time.values)[:13]} + {gda.fhr.values}h")
        else:
            axs[0].set(title=f"GraphUFS, {str(gda.time.values)[:13]}")
        axs[1].set(title=f"ERA5, {str(date)[:13]}");
        fig.colorbar(p, ax=axs, orientation="horizontal", shrink=.6, aspect=35, label=label, extend=extend)
        return fig, axs


def get_extend(xdslist, vmin, vmax):
    minval = np.min([xds.min().values for xds in xdslist])
    maxval = np.max([xds.max().values for xds in xdslist])
    vmin = minval if vmin is None else vmin
    vmax = maxval if vmax is None else vmax

    extend = "neither"
    if minval < vmin:
        extend = "min"
    if maxval > vmax:
        extend = "max" if extend == "neither" else "both"

    return extend, vmin, vmax
