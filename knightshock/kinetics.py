from typing import Type

import cantera as ct
import numpy as np


class Simulation:
    """
    A class for initializing and running a zero-dimensional homogeneous reactor simulation.

    Attributes:
        gas: Cantera `Solution` object.
        reactor: Cantera `Reactor` object.
        reactor_net: Cantera `ReactorNet` object.
        states: Cantera `SolutionArray` object.

    """

    def __init__(
        self,
        gas: ct.Solution | str,
        T: float,
        P: float,
        X: str | dict[str, float],
        *,
        reactor: ct.Reactor | Type[ct.Reactor] = ct.Reactor,
    ):
        """

        Args:
            gas: Cantera gas phase object or filepath to mechanism.
            T: Temperature [K].
            P: Pressure [Pa].
            X: Species mole fractions.
            reactor: Cantera reactor object or subclass (optional).

        """

        self.gas = gas if isinstance(gas, ct.Solution) else ct.Solution(gas)
        self.gas.TPX = T, P, X

        try:
            # ct.Reactor is considered to be a subclass of itself
            if isinstance(reactor, ct.Reactor):
                self.reactor = reactor
                self.reactor.insert(self.gas)

            # Raises TypeError if argument is not a class
            elif issubclass(reactor, ct.Reactor):
                self.reactor = reactor(self.gas)

            else:
                raise TypeError

        except TypeError:
            raise TypeError(
                "Reactor argument must be a ct.Reactor object or subclass."
            ) from None

        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(self.gas, extra=["t"])

        self.states.append(self.reactor.thermo.state, t=0)  # Add initial state

    def run(
        self,
        t: float = 10e-3,
    ):
        """
        Args:
            t: Simulation end time [s] (optional).

        """
        i = 0
        while self.reactor_net.time < t:
            self.reactor_net.step()
            self.states.append(self.reactor.thermo.state, t=self.reactor_net.time)
            i += 1

        return self

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

    def ignition_delay_time(
        self, species: str = None, *, method: str = "inflection"
    ) -> float:
        """
        Calculates the ignition delay time from the reactor temperature history, or species mole fraction if given,
        using the specified method.

        !!! Note
            Returns [`np.nan`](https://numpy.org/doc/stable/reference/constants.html#numpy.nan) if calculated
            ignition delay time occurs at the end of the simulated time.

        Args:
            species: Name of species (optional).
            method:
                Method used to calculate ignition delay time (optional).

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
            raise ValueError(
                f"Invalid method '{method}'; valid methods are 'inflection' and 'peak'."
            )

    def get_top_species(
        self, n: int = None, *, exclude: str | list[str] = None
    ) -> list[str]:
        """
        Returns the top `n` species by mole fraction in descending order. If `n` is not given,
        all non-excluded species are returned.

        Args:
            n: Number of species (optional).
            exclude: Species to exclude (optional).

        Returns:
            List of top species.

        """

        X_max = np.max(self.states.X.T, axis=1)
        species = [
            t[1] for t in sorted(zip(X_max, self.states.species_names), reverse=True)
        ]

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for s in exclude:
                try:
                    species.remove(s.upper())
                except ValueError:
                    pass

        return species[:n]
