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
        IDT =sim.ignition_delay_time(self.IDT_species, method="peak") * 1e6
        print(f'saving {str(sim.gas.source).split(".")[0]}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}')
        # 'Saving species time history data'
        # Xhistory = pd.DataFrame()
        # Xhistory['t'] = sim.t
        # Xhistory['T'] = sim.P
        # Xhistory['P'] = sim.P
        # for species in sim.get_top_species():
        #     species_history = pd.DataFrame(sim.X(species), columns=[species])
        #     Xhistory = pd.concat([Xhistory, species_history], axis=1)
        # Xhistory.to_csv(f'{os.path.basename(sim.gas.source)}_P={P / 1e6:.2}MPa_T={T}K_{mix_name}.csv')
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
    T = np.linspace(1000, 1500, num=25)
    P = np.array([25e5])
    mixtures = {}
    mixtures['mix 1'] = 'CH4: 0.091184356, C2H6: 0.002106931, C3H8: 0.000072104, C4H10: 0.000047383, CO2: 0.000046633, O2: 0.190411476, N2: 0.715969811'
    # mixtures['mix 2'] = 'CH4: 0.069262296, C2H6: 0.001600394, C3H8: 0.000054769, C4H10: 0.000035991, H2: 0.071128611, CO2: 0.000035422, O2: 0.180198063, N2: 0.677716846'
    # mixtures['mix 3'] = 'CH4: 0.063857226, C2H6: 0.001475503, C3H8: 0.000050495, C4H10: 0.000033182, CO2: 0.000032658, NH3: 0.065577898, O2: 0.182530295, N2: 0.686329779'
    mixtures['mix 4'] = 'CH4: 0.047831691, C2H6: 0.001105212, C3H8: 0.000037823, C4H10: 0.000024855, CO2: 0.000024462, O2: 0.199764591, N2: 0.75114864'
    # mixtures['mix 5'] = 'CH4: 0.037283039, C2H6: 0.000861472, C3H8: 0.000029481, C4H10: 0.000019374, CO2: 0.000019067, H2: 0.038287653, O2: 0.193996784, N2: 0.729520565'
    # mixtures['mix 6'] = 'CH4: 0.03416936, C2H6: 0.00078952, C3H8: 0.00002701, C4H10: 0.00001775, CO2: 0.00001747, NH3: 0.035090083, O2: 0.195340301, N2: 0.73456445'
    # mixtures['mix 7'] = 'CH4: 0.1212762678, C2H6: 0.00280216, C3H8: 0.000095896, C4H10: 0.000063017, CO2: 0.000062021, H2: 0.124540444, O2: 0.157756116, N2: 0.593464383'
    # mixtures['mix 8'] = 'CH4: 0.16675, C2H6: 0.00385, C3H8: 0.00013, C4H10: 0.00008, CO2: 0.00008, O2: 0.17410, N2: 0.65505'
    # mixtures['mix 9'] = 'CH4: 0.112906155, C2H6: 0.002608841, C3H8: 0.00008928, C4H10: 0.00005867, CO2: 0.000057742, NH3: 0.115948481, O2: 0.161366185, N2: 0.6067010595'
    Mechanisms = [r'C:\Users\mpier\GitHub\KnightShock_IDTsims\NUIGMech12_modified.yaml']
    # Mechanisms = [r'C:\Users\mpier\GitHub\KnightShock_IDTsims\aramco3.yaml']

    t1 = time()
    SimulationPool(Mechanisms, mixtures, P, T, IDT_species='OH').parallel()
    t2 = time()
    print('Parallel: {0:.3f} seconds'.format(t2 - t1))

    # t1 = time()
    # SimulationPool(Mechanisms, mixtures, P, T).serial()
    # t2 = time()
    # print('Serial: {0:.3f} seconds'.format(t2 - t1))
