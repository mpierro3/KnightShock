import numpy as np
import numpy.typing as npt
import matplotlib as mpl
from matplotlib import pyplot as plt


class IDTFigure:
    """Class for creating IDT figures with the standard layout:

    - Inverse temperature (1000/T) x-axis (bottom)
    - Log-scale IDT y-axis
    - Secondary temperature x-axis (top)

    Attributes:
        ax: Inverse temperature axis.
        ax2: Temperature axis.

    """

    exp_props = {"linestyle": "", "marker": "o", "capsize": 5}
    """Default properties for all experimental error bars."""

    sim_props = {}
    """Default properties for all simulation lines."""

    def __init__(self, ax: mpl.axes.Axes | None = None):
        """

        Args:
            ax: Matplotlib [`Axes`](https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes)
                object for plotting (optional)

        """

        if ax is None:
            _, self.ax = plt.subplots()

        def convert(x):
            return 1000 / x

        self.ax.set_yscale("log")
        self.ax2 = self.ax.secondary_xaxis('top', functions=(convert, convert))

        self.ax.set_ylabel(f"Ignition Delay Time [$μs$]")
        self.ax.set_xlabel("1000/T [$K^-1$]")
        self.ax2.set_xlabel("Temperature [$K$]")

        self.ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1.25)))
        self.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0f}"))

    def add_exp(
            self,
            T: int | float | npt.ArrayLike,
            IDT: int | float | npt.ArrayLike,
            uncertainty: float = 0,
            **kwargs
    ):
        """Add experimental ignition delay data with uncertainty to plot.

        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [μs].
            uncertainty: Experimental uncertainty.

        """
        T = np.asarray(T)
        IDT = np.asarray(IDT)
        return self.ax.errorbar(1000 / T, IDT, yerr=uncertainty * IDT, **(self.exp_props | kwargs))

    def add_sim(
            self,
            T: int | float | npt.ArrayLike,
            IDT: int | float | npt.ArrayLike,
            **kwargs
    ) -> list[mpl.lines.Line2D]:
        """Add simulated ignition delay data to plot.

        Args:
            T: Temperatures [K].
            IDT: Ignition delay times [μs].

        """
        T = np.asarray(T)
        return self.ax.plot(1000 / T, IDT, **(self.sim_props | kwargs))

    @property
    def T_lim(self) -> tuple[float, float]:
        value = self.ax.get_xlim()
        return 1000 / value[1], 1000 / value[0]

    @T_lim.setter
    def T_lim(self, value: tuple[float, float]):
        self.ax.set_xlim(1000 / value[1], 1000 / value[0])

    @property
    def IDT_lim(self) -> tuple[float, float]:
        return self.ax.get_ylim()

    @IDT_lim.setter
    def IDT_lim(self, value: tuple[float, float]):
        self.ax.set_ylim(value)
