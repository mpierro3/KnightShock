from shocktube.kinetics import SimulationPool
from shocktube.figures import IDTPlot

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    sim_pool = SimulationPool(
        np.linspace(1100, 1300, 100),
        200e5,
        "CH4: 0.05, O2: 0.10, AR: 0.85",
        "gri30.yaml"
    )

    plot = IDTPlot()
    plot.add_sim(sim_pool.cases["T"].values, sim_pool.cases["IDT"].values * 1E6)
    plt.show()
