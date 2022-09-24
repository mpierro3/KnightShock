"""
Absorption spectroscopy module
"""

import numpy.typing as npt


AVOGADRO_NUMBER = 6.02214076e23
"""Avogadro's number [molecules/mol]."""

GAS_CONSTANT = 8.31432e3
"""Universal gas constant [J mol^-1^ K^-1^]."""


def species_mole_fraction(
        A: float | npt.NDArray[float],
        sigma: float | npt.NDArray[float],
        T: float | npt.NDArray[float],
        P: float | npt.NDArray[float],
        L: float | npt.NDArray[float]
) -> float | npt.NDArray[float]:
    r"""
    Calculates the species mole fraction according to the Beer-Lambert law

    $$
    A = \sigma X N_A L \frac{P}{RT}
    $$

    Parameters
    ----------
    A:
        Absorbance.
    sigma:
        Absorption cross-section [cm^2^].
    T:
        Absolute temperature [K].
    P:
        Absolute pressure [Pa].
    L:
        Path length [cm].

    Returns
    -------
    X:
        Species mole fraction.

    """

    return A / (sigma / 1E6 * AVOGADRO_NUMBER * L) * (GAS_CONSTANT * T) / P
