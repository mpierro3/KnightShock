import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker


class IDTPlot:
    """
    Attributes:
        ax: Inverted temperature axis.
        ax2: Temperature axis.
    """

    exp_props = {"linestyle": "", "marker": "o"}
    """Default properties for all experimental error bars."""

    sim_props = {}
    """Default properties for all simulation lines."""

    def __init__(self, ax=None):
        if ax is None:
            _, self.ax = plt.subplots()

        def convert(x):
            return 1000 / x

        self.ax.set_yscale("log")
        self.ax2 = self.ax.secondary_xaxis('top', functions=(convert, convert))

        self.ax.set_ylabel(f"Ignition Delay Time [μs]")
        self.ax.set_xlabel("1000/T [1/K]")
        self.ax2.set_xlabel("Temperature [K]")

        self.ax.yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1.25)))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    def add_exp(self, T, tau, uncertainty=0, **args):
        """Add experimental ignition delay data with uncertainty to plot."""
        T = np.asarray(T)
        tau = np.asarray(tau)
        return self.ax.errorbar(1000 / T, tau, yerr=uncertainty * tau, **(IDTPlot.exp_props | args))

    def add_sim(self, T, tau, **args):
        """Add simulated ignition delay data to plot."""
        T = np.asarray(T)
        return self.ax.plot(1000 / T, tau, **(IDTPlot.sim_props | args))

    @property
    def T_lim(self) -> tuple[float, float]:
        value = self.ax.get_xlim()
        return 1000 / value[1], 1000 / value[0]

    @T_lim.setter
    def T_lim(self, value: tuple[float, float]):
        self.ax.set_xlim(1000 / value[1], 1000 / value[0])

    @property
    def tau_lim(self) -> tuple[float, float]:
        return self.ax.get_ylim()

    @tau_lim.setter
    def tau_lim(self, value: tuple[float, float]):
        self.ax.set_ylim(value)
