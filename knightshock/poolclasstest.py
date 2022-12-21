import multiprocessing
import numpy as np
import cantera as ct
import itertools
from time import time
from kinetics import Simulation
import pandas as pd


class SimulationPool:

    def __init__(self, Mechs: list, Mixtures: dict, Pressures: np.array, Temps: np.array, *, nProcs=None):

        self.Mechs = Mechs
        self.gases = {}
        self.Mixtures = Mixtures
        self.initialPressures = Pressures
        self.intialTemps = Temps
        self.parallel()

    def parallel(self):
        """
        Call the function ``run`` on ``Args`` processors for ``number of processors``
        """
        tupled_inputs = list(itertools.product(self.Mechs, self.Mixtures.values(),
                                               self.initialPressures, self.intialTemps))

        with multiprocessing.Pool(
                # processes=nProcs, initializer=init_process, initargs=(mechs,)
        ) as pool:
            y = pool.map(self.run, tupled_inputs)

        return y

    def init_process(self, mech):
        """
        This function is called once for each process in the Pool. We use it to
        initialize any Cantera objects we need to use
        """

        self.gases[mech] = ct.Solution(mech)
        self.gases[mech].transport_model = 'Multi'

    def run(self, args):
        mech, X, P, T = args
        self.init_process(mech)
        gas = self.gases[mech]
        print(f'Running Cantera {ct.__version__}  {gas.source}, {T = },{P = },{X = }')
        gas.TPX = T, P, X
        sim = Simulation(gas, T, P, X)
        sim.run()

        self.save_to_CSV(sim, args)
        return

    def save_to_CSV(self, sim, initialconditions):
        # Current work in progress
        """function for saving simulation data to csv"""
        mech, X, P, T = initialconditions
        mix_name = self.get_key(self.Mixtures, X)
        print(f'{mech}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}')
        Xhistory = pd.DataFrame()
        for species in sim.get_top_species():
            Xhistory[species] = sim.X(species)
        Xhistory.to_csv(f'{mech}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}.csv')

    def get_key(self, dictionary, val):
        for key, value in dictionary.items():
            if val == value:
                return key
        return "key doesn't exist"


if __name__ == '__main__':
    T = np.array([1300])
    P = np.array([1e6])
    mixtures = {}
    mixtures['mix A'] = 'CH4: 0.04, O2: 0.08, AR: 0.88'
    mixtures['mix B'] = 'CH4: 0.05, O2: 0.08, AR: 0.88'
    Mechanisms = ['aramco3.yaml']

    SimulationPool(Mechanisms, mixtures, P, T)
