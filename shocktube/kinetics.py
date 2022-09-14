import cantera as ct
import numpy as np


class Simulation:
    def __init__(self, gas: ct.Solution, T: float, P: float, X, t: float = 5e-3):
        """
        A class for initializing and running a zero-dimensional homogeneous reactor simulation.

        Parameters:
            gas: Cantera gas phase object.
            T: Temperature [K].
            P: Pressure [Pa].
            X: Species mole fractions.
            t: Time [s].

        """

        gas.TPX = T, P, X
        self.reactor = ct.Reactor(gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(gas, extra=["t"])

        self.states.append(self.reactor.thermo.state, t=0)  # Add initial state

        i = 0
        while self.reactor_net.time < t:
            self.reactor_net.step()
            self.states.append(self.reactor.thermo.state, t=self.reactor_net.time)
            i += 1

    @property
    def t(self):
        return self.states.t

    @property
    def T(self):
        return self.states.T

    @property
    def P(self):
        return self.states.P

    def X(self, species: str):
        return self.states(species).X.flatten()

    def ignition_delay_time(self, species: str = None, *, method: str = "inflection") -> float:
        """
        Calculates the ignition delay time from the reactor temperature, or `species` mole fraction if given,
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
            return self.t[np.argmax(np.diff(x) / np.diff(self.t))]
        elif method == "peak":
            return self.t[np.argmax(x)]
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

