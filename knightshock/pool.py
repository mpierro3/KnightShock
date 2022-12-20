import multiprocessing
import numpy as np
import cantera as ct
import itertools
from time import time
from kinetics import Simulation
import pandas as pd

# Global storage for Cantera Solution objects
gases = {}


def save_to_CSV(sim: Simulation):
    #Current work in progress
    'function for saving simulation data to csv'
    print(sim.get_top_species(5))
    pass


def init_process(mech):
    """
    This function is called once for each process in the Pool. We use it to
    initialize any Cantera objects we need to use
    """
    gases[mech] = ct.Solution(mech)
    gases[mech].transport_model = 'Multi'


def run(args):
    mech, T, P, X = args
    print(f'Running Cantera {ct.__version__}  {T = },{P = },{X = }')
    gas = gases[mech]
    gas.TPX = T, P, X
    sim = Simulation(gas, T, P, X)
    sim.run()
    save_to_CSV(sim)

    return


def parallel(mech, Pressure, Temps, mix, *, nProcs=None):
    """
    Call the function ``run`` on ``Args`` processors for ``number of processors``
    """
    with multiprocessing.Pool(
            processes=nProcs, initializer=init_process, initargs=(mech,)
    ) as pool:
        y = pool.map(run,
                     zip(itertools.repeat(mech),
                         Temps,
                         itertools.repeat(Pressure),
                         itertools.repeat(mix)))

    return


def serial(mech, Pressure, Temps, mix):
    init_process(mech)
    y = list(map(run,
                 zip(itertools.repeat(mech),
                     Temps,
                     itertools.repeat(Pressure),
                     itertools.repeat(mix))))
    return y


if __name__ == '__main__':

    T = np.array([1100, 1200, 1300, 1400, 1500])
    P = 2e6

    # It seems like serial is faster for smaller mechanisms the break even point it about ~4 for aramco3

    X = 'CH4: 0.04, O2: 0.08, AR: 0.88'
    t1 = time()
    parallel('gri30.yaml', P, T, X)
    t2 = time()
    print('Parallel: {0:.3f} seconds'.format(t2 - t1))

    t1 = time()
    serial('gri30.yaml', P, T, X)
    t2 = time()
    print('Serial: {0:.3f} seconds'.format(t2 - t1))
