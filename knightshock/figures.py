import numpy as np
import numpy.typing as npt
import matplotlib as mpl
from matplotlib import pyplot as plt


class IDTFigure:
    """Class for creating ignition delay time figures with the standard layout:

    - Inverse temperature (1000/T) x-axis (bottom)
    - Log-scale IDT y-axis
    - Secondary temperature x-axis (top)

    Functions are provided for plotting experimental data ([`add_exp`][knightshock.figures.IDTFigure.add_exp]) as
    scatter plots with or without error bars and for plotting model predictions from simulations
    ([`add_sim`][knightshock.figures.IDTFigure.add_sim]) as line plots, in addition to other functionality.

    !!! Important
        Keyword arguments to functions are passed to the underlying matplotlib function calls and override
        the default property dicts set as class attributes.

    Attributes:
        ax: Inverse temperature axis.
        ax2: Temperature axis.

    """

    exp_props = {"linestyle": "None", "marker": "o"}
    """Default properties for all experimental scatter (including errorbar) plots."""

    error_props = {"capsize": 5}
    """Default properties for errorbar plots."""

    sim_props = {}
    """Default properties for all simulation line plots."""

    units: str = "μs"
    """Default units for ignition delay time."""

    def __init__(self, ax: mpl.axes.Axes | None = None):
        """
        Args:
            ax: Existing matplotlib [`Axes`](https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes)
                object for plotting (optional).

        """

        if ax is None:
            _, self.ax = plt.subplots()

        def convert(x):
            return 1000 / x

        self.ax.set_yscale("log")
        self.ax2 = self.ax.secondary_xaxis("top", functions=(convert, convert))

        self.exp_handles = []
        self.exp_labels = []
        self.sim_handles = []
        self.sim_labels = []

        self.ax.set_ylabel(r"Ignition Delay Time [$\mathrm{" + self.units + "}$]")
        self.ax.set_xlabel(r"1000/T [$\mathrm{K^{-1}}$]")
        self.ax2.set_xlabel(r"Temperature [$\mathrm{K}$]")

        self.ax.yaxis.set_minor_formatter(
            mpl.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1.25))
        )
        self.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0f}"))

    def add_exp(
        self,
        T: int | float | npt.ArrayLike,
        IDT: int | float | npt.ArrayLike,
        uncertainty: float = 0,
        **kwargs
    ) -> mpl.collections.PathCollection | mpl.container.ErrorbarContainer:
        """
        Add experimental ignition delay data to the figure as scatter plots. If `uncertainty` is given,
        error bars are included.

        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [[`units`][knightshock.figures.IDTFigure.units]].
            uncertainty: Experimental uncertainty as a fraction of `IDT` (optional).

        """
        T = np.asarray(T)
        IDT = np.asarray(IDT)

        if uncertainty == 0:
            c = self.ax.scatter(1000 / T, IDT, **(self.exp_props | kwargs))
            self.exp_handles.append(c)
        else:
            c = self.ax.errorbar(
                1000 / T,
                IDT,
                yerr=uncertainty * IDT,
                **(self.exp_props | self.error_props | kwargs)
            )
            self.exp_handles.append(c[0])

        self.exp_labels.append(kwargs["label"] if "label" in kwargs else None)
        return c

    def add_sim(
        self, T: int | float | npt.ArrayLike, IDT: int | float | npt.ArrayLike, **kwargs
    ) -> list[mpl.lines.Line2D]:
        """Add ignition delay model predictions to the figure.

        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [[`units`][knightshock.figures.IDTFigure.units]].

        """
        T = np.asarray(T)
        (ln,) = self.ax.plot(1000 / T, IDT, **(self.sim_props | kwargs))

        self.sim_handles.append(ln)
        self.sim_labels.append(kwargs["label"] if "label" in kwargs else None)

        return ln

    def legend(self, **kwargs) -> mpl.legend.Legend:
        """
        Create a legend for all data adding to the figure using [`add_exp`][knightshock.figures.IDTFigure.add_exp] or
        [`add_sim`][knightshock.figures.IDTFigure.add_sim].
        """
        return self.ax.legend(
            handles=[
                h for h, l in zip(self.sim_handles, self.sim_labels) if l is not None
            ]
            + [h for h, l in zip(self.exp_handles, self.exp_labels) if l is not None],
            labels=[l for l in self.sim_labels if l is not None]
            + [l for l in self.exp_labels if l is not None],
            **kwargs
        )

    @property
    def T_lim(self) -> tuple[float, float]:
        """Get/set the temperature [K] limits of the figure."""
        value = self.ax.get_xlim()
        return 1000 / value[1], 1000 / value[0]

    @T_lim.setter
    def T_lim(self, value: tuple[float, float]):
        self.ax.set_xlim(1000 / value[1], 1000 / value[0])

    @property
    def IDT_lim(self) -> tuple[float, float]:
        """Get/set the ignition delay time [[`units`][knightshock.figures.IDTFigure.units]] limits of the figure."""
        return self.ax.get_ylim()

    @IDT_lim.setter
    def IDT_lim(self, value: tuple[float, float]):
        self.ax.set_ylim(value)
