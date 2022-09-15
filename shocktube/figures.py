from matplotlib import pyplot as plt
from matplotlib import ticker


class IDTPlot:
    """
    Attributes:
        ax: Inverted temperature axis.
        ax2: Temperature axis.
    """

    exp_props = {"linestyle": "", "marker": "o"}
    sim_props = {}

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

    def add_exp(self, T, IDT, uncertainty=0, **args):
        """Add experimental ignition delay data with uncertainty to plot."""
        self.ax.errorbar(1000 / T, IDT, yerr=uncertainty*IDT, **(IDTPlot.exp_props | args))

    def add_sim(self, T, IDT, **args):
        """Add simulated ignition delay data to plot."""
        self.ax.plot(1000 / T, IDT, **(IDTPlot.sim_props | args))

    @property
    def T_limits(self) -> tuple[float, float]:
        value = self.ax.get_xlim()
        return 1000 / value[1], 1000 / value[0]

    @T_limits.setter
    def T_limits(self, value: tuple[float, float]):
        self.ax.set_xlim(1000 / value[1], 1000 / value[0])

    @property
    def IDT_limits(self) -> tuple[float, float]:
        return self.ax.get_ylim()

    @IDT_limits.setter
    def IDT_limits(self, value: tuple[float, float]):
        self.ax.set_ylim(value)