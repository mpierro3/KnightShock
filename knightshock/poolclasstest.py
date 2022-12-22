import multiprocessing
import numpy as np
import cantera as ct
import itertools
from time import time
from kinetics import Simulation
import pandas as pd
import csv
import os


class SimulationPool:

    def __init__(self, Mechs: list, Mixtures: dict, Pressures: np.array, Temps: np.array, *,IDT_species=None, nProcs=None):

        self.Mechs = Mechs
        self.gases = {}
        self.Mixtures = Mixtures
        self.initialPressures = Pressures
        self.intialTemps = Temps
        self.IDT_species =IDT_species

        for mix in self.Mixtures.keys():
            with open(f'{mix} IDT.csv', 'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['mechanism','Pressure(Pa)', 'Temperature(K)', 'IDT(s)'])

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
        IDT =sim.ignition_delay_time(self.IDT_species)
        print(f'saving {str(sim.gas.source).split(".")[0]}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}')
        'Saving species time history data'
        Xhistory = pd.DataFrame()
        Xhistory['t'] = sim.t
        Xhistory['T'] = sim.P
        Xhistory['P'] = sim.P
        for species in sim.get_top_species():
            species_history = pd.DataFrame(sim.X(species), columns=[species])
            Xhistory = pd.concat([Xhistory, species_history], axis=1)
        Xhistory.to_csv(f'{os.path.basename(sim.gas.source)}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}.csv')
        'Saving IDT data'
        with open(f'{mix_name} IDT.csv', 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sim.gas.source,P,T,IDT])

    def get_key(self, dictionary, val):
        for key, value in dictionary.items():
            if val == value:
                return key
        return "key doesn't exist"

    def serial(self):
        tupled_inputs = list(itertools.product(self.Mechs, self.Mixtures.values(),
                                               self.initialPressures, self.intialTemps))

        y = list(map(self.run, tupled_inputs))
        return y


if __name__ == '__main__':
    T = np.array([1300])
    P = np.array([1e6])
    mixtures = {}

    mixtures['mix A'] = 'CH4: 0.04, O2: 0.08, AR: 0.88'
    Mechanisms = [r'C:\Users\ch733787\Documents\GitHub\KnightShock\HPST\gri30.yaml']

    t1 = time()
    SimulationPool(Mechanisms, mixtures, P, T,IDT_species='OH').parallel()
    t2 = time()
    print('Parallel: {0:.3f} seconds'.format(t2 - t1))

    # t1 = time()
    # SimulationPool(Mechanisms, mixtures, P, T).serial()
    # t2 = time()
    # print('Serial: {0:.3f} seconds'.format(t2 - t1))
