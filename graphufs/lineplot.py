import matplotlib.pyplot as plt

class LinePlotter():

    @staticmethod
    def subplots(nrows, ncols, **kwargs):
        return plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*4), constrained_layout=True)

    @staticmethod
    def title(fldname):
        return " ".join([x.capitalize() if x != "of" else x for x in fldname.replace("_", " ").split(" ")])

    @staticmethod
    def nicefig(fig, metric, truth):
        fig.legend(loc="center left", bbox_to_anchor=(1, .5))
        mname = metric.upper() if metric != "bias" else "Bias"
        fig.suptitle(f"{mname} vs {truth}")

    def plot_surface(self, dsdict: dict, truth: str, metric: str = "mae", fields: tuple[str] = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind")):

        fig, axs = self.subplots(1, len(fields))

        self._plot_row(axs, dsdict, truth, metric, fields)
        self.nicefig(fig, metric, truth)
        return fig, axs

    def plot_levels(
        self,
        dsdict: dict,
        truth: str,
        metric: str = "mae",
        fields: tuple[str] = ("temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"),
        levels: tuple[int] = (100, 500, 850),
    ):

        fig, axs = self.subplots(len(levels), len(fields))

        if len(levels) == 1 or len(fields) == 1:
            axs = [axs]

        for level, axr in zip(levels, axs):
            self._plot_row(axr, dsdict, truth, metric, fields, level=level)

        self.nicefig(fig, metric, truth)
        return fig, axs


    def _plot_row(self, axr, dsdict, truth, metric, fields, level=None):

        xticks = dsdict["GraphUFS"][truth].fhr.values[3::4]

        for fld, ax in zip(fields, axr):
            sps = ax.get_subplotspec()
            for j, (label, xdict) in enumerate(dsdict.items()):
                xds = xdict[truth]
                if label == "Replay":
                    color="gray"
                elif label == "Replay Targets":
                    color = "C5"
                else:
                    color = f"C{j}"

                kw = {
                    "color": color,
                    "label": label if sps.is_last_row() and sps.is_first_col() else None
                }
                if fld in xds:
                    plotme = xds[fld].sel(metric=metric)
                    if level is not None:
                        if level in plotme.level:
                            plotme = plotme.sel(level=level)
                        else:
                            plotme = None

                    if plotme is not None:
                        if "lead_time" in plotme.coords:
                            plotme.plot(ax=ax, **kw)
                        else:
                            ax.axhline(plotme, **kw)

            ax.set(
                xlabel="Lead time (days)" if sps.is_last_row() else "",
                ylabel="" if level is None or not sps.is_first_col() else f"{level} hPa",
                title=self.title(fld) if sps.is_first_row() else "",
                xticks=xticks,
                xticklabels=[x//24 for x in xticks],
                ylim=[0, None] if metric in ("mae", "mse") else [None, None],
                xlim=[-6, 252],
            )
