from multiprocessing import Pool
from itertools import product
from typing import Iterable

import cantera as ct
import numpy as np
import pandas as pd


class Simulation:
    def __init__(self, gas: ct.Solution | str, T: float, P: float, X, t: float = 5e-3):
        """
        A class for initializing and running a zero-dimensional homogeneous reactor simulation.

        Parameters:
            gas: Cantera gas phase object or filepath to mechanism.
            T: Temperature [K].
            P: Pressure [Pa].
            X: Species mole fractions.
            t: End time [s].

        """

        gas = gas if isinstance(gas, ct.Solution) else ct.Solution(gas)

        gas.TPX = T, P, X
        self.reactor = ct.Reactor(gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(gas, extra=["t"])

        self.states.append(self.reactor.thermo.state, t=0)  # Add initial state

        i = 0
        while self.reactor_net.time < t:
            self.reactor_net.step()
            self.states.append(self.reactor.thermo.state, t=self.reactor_net.time)
            if i % 100 == 0:
                print(f"t = {self.reactor_net.time * 1e6:.1f} μs")
            i += 1

    @property
    def t(self) -> np.ndarray[float]:
        """Reactor elapsed time [s]."""
        return self.states.t

    @property
    def T(self) -> np.ndarray[float]:
        """Reactor temperature history [K]."""
        return self.states.T

    @property
    def P(self) -> np.ndarray[float]:
        """Reactor pressure history [Pa]."""
        return self.states.P

    def X(self, species: str) -> np.ndarray[float]:
        """
        Returns the mole fraction history for the given species.

        Args:
            species: Name of species.

        """
        return self.states(species).X.flatten()

    def ignition_delay_time(self, species: str = None, *, method: str = "inflection") -> float:
        """
        Calculates the ignition delay time from the reactor temperature history, or `species` mole fraction if given,
        using the specified `method`.

        Args:
            species: Name of species.
            method:
                Method used to calculate ignition delay time.
                  - 'inflection' point (max slope)
                  - 'peak'

        Returns:
            Ignition delay time [s].

        """

        x = self.T if species is None else self.X(species)
        if method == "inflection":
            i = np.argmax(np.diff(x) / np.diff(self.t))
            return self.t[i] if i != len(self.t) - 2 else np.nan
        elif method == "peak":
            i = np.argmax(x)
            return self.t[i] if i != len(self.t) - 1 else np.nan
        else:
            raise ValueError(f"Invalid method '{method}'; valid methods are 'inflection' and 'peak'.")

    def get_top_species(self, n: int = None, *, exclude: str | list[str] = None) -> list[str]:
        """
        Returns the top `n` species by mole fraction in descending order. If `n` is not given,
        all non-excluded species are returned.

        Args:
            n: Number of species.
            exclude: Species to exclude.

        Returns:
            List of top species.

        """

        X_max = np.max(self.states.X.T, axis=1)
        species = [t[1] for t in sorted(zip(X_max, self.states.species_names), reverse=True)]

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for s in exclude:
                try:
                    species.remove(s.upper())
                except ValueError:
                    pass

        return species[:n]


class SimulationPool:
    def __init__(self, mech, T, P, X, n: int = None, **kwargs):
        self.cases = pd.DataFrame({"mech": mech, "T": T, "P": P, "X": X})

        def run_case(*args):
            return Simulation(*args).ignition_delay_time(**kwargs)

        with Pool(n) as pool:
            self.cases["tau"] = pool.starmap(run_case, self)

    def __getitem__(self, item):
        return self.cases.iloc[item].values

    @staticmethod
    def _run_case(T: float, P: float, X, mechanism: str, **kwargs):
        gas = ct.Solution(mechanism)
        sim = Simulation(gas, T, P, X)
        return sim.ignition_delay_time(**kwargs)

    @classmethod
    def parameter_study(
            cls,
            mech: str | list[str],
            T: float | Iterable[float],
            P: float | Iterable[float],
            X: str | dict[str, float] | list[str | dict[str, float]],
            *args,
            **kwargs
    ):
        if isinstance(T, float):
            T = [T]
        if isinstance(P, float):
            P = [P]
        if isinstance(X, (str, dict)):
            X = [X]
        if isinstance(mech, str):
            mech = [mech]

        return cls(*zip(*product(
            list(mech),
            list(T),
            list(P),
            list(X),
        )), *args, **kwargs)
