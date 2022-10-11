import numpy.typing as npt


AVOGADRO_NUMBER = 6.02214076e23
"""Avogadro's number [molecules/mol]."""

GAS_CONSTANT = 8.31432e3
"""Universal gas constant [J mol^-1^ K^-1^]."""


def absorbance(
        X: float | npt.NDArray[float],
        sigma: float | npt.NDArray[float],
        T: float | npt.NDArray[float],
        P: float | npt.NDArray[float],
        L: float | npt.NDArray[float]
) -> float | npt.NDArray[float]:
    r"""Calculates the absorption for a species from the Beer-Lambert law.

    Args:
        X: Species mole fraction.
        sigma: Absorption cross-section [cm^2^].
        T: Absolute temperature [K].
        P: Absolute pressure [Pa].
        L: Path length [cm].

    Returns:
        A: Absorbance.

    """
    return sigma * X * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T) / 1E6


def absorption_cross_section(
        A: float | npt.NDArray[float],
        X: float | npt.NDArray[float],
        T: float | npt.NDArray[float],
        P: float | npt.NDArray[float],
        L: float | npt.NDArray[float]
) -> float | npt.NDArray[float]:
    """Calculates the species mole fraction from the Beer-Lambert law.

    Args:
        A: Absorbance.
        X: Species mole fraction.
        T: Absolute temperature [K].
        P: Absolute pressure [Pa].
        L: Path length [cm].

    Returns:
        sigma: Absorption cross-section [cm^2^].

    """

    return A / (X * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T)) * 1E6


def species_mole_fraction(
        A: float | npt.NDArray[float],
        sigma: float | npt.NDArray[float],
        T: float | npt.NDArray[float],
        P: float | npt.NDArray[float],
        L: float | npt.NDArray[float]
) -> float | npt.NDArray[float]:
    """Calculates the species mole fraction from the Beer-Lambert law.

    Args:
        A: Absorbance.
        sigma: Absorption cross-section [cm^2^].
        T: Absolute temperature [K].
        P: Absolute pressure [Pa].
        L: Path length [cm].

    Returns:
        X: Species mole fraction.

    """

    return A / (sigma * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T)) * 1E6
