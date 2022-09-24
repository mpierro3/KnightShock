import knightshock as ks


from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    sim_pool = ks.kinetics.SimulationPool(
        "gri30.yaml",
        np.linspace(1000, 1300, 100),
        200e5,
        {"CH4": 0.05, "O2": 0.10, "AR": 0.85}
    )

    plot = ks.figures.IDTPlot()
    plot.add_sim(sim_pool.cases["T"].values, sim_pool.cases["tau"].values * 1E6)
    plt.show()
