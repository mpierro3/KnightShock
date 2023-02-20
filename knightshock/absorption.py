from .constants import AVOGADRO_NUMBER, GAS_CONSTANT

import numpy as np
import numpy.typing as npt


def absorbance(
    X: float | npt.NDArray[float],
    sigma: float | npt.NDArray[float],
    T: float | npt.NDArray[float],
    P: float | npt.NDArray[float],
    L: float = 1,
) -> float | npt.NDArray[float]:
    r"""Calculates the absorption for a species from the Beer-Lambert law.

    Args:
        X: Species mole fraction.
        sigma: Absorption cross-section [cm^2^].
        T: Absolute temperature [K].
        P: Absolute pressure [Pa].
        L: Path length [cm] (optional).

    Returns:
        A: Absorbance.

    """
    return sigma * X * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T) / 1e6


def absorption_cross_section(
    A: float | npt.NDArray[float],
    X: float | npt.NDArray[float],
    T: float | npt.NDArray[float],
    P: float | npt.NDArray[float],
    L: float,
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

    return A / (X * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T)) * 1e6


def species_mole_fraction(
    A: float | npt.NDArray[float],
    sigma: float | npt.NDArray[float],
    T: float | npt.NDArray[float],
    P: float | npt.NDArray[float],
    L: float,
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

    return A / (sigma * AVOGADRO_NUMBER * L * P / (GAS_CONSTANT * T)) * 1e6


def multi_species_mole_fraction(
    A: npt.NDArray[float],
    sigma: npt.NDArray[float],
    T: float | npt.NDArray[float],
    P: float | npt.NDArray[float],
    L: float,
) -> npt.NDArray[float]:
    r"""
    :fontawesome-solid-flask: Experimental

    Calculates the mole fractions of `N` species from absorbance data at `N` wavelengths given absorption
    cross-sections for each species at each wavelength. For mole fraction time histories, absorption cross-sections,
    temperature, and pressure can be constant or varying with time.

    Args:
        A: Absorbance at each wavelength `(N, ...)`.
        sigma: Species absorption cross-sections [cm^2] at each wavelength `(N, N, ...)`.
        T: Absolute temperature [K].
        P: Absolute pressure [Pa].
        L: Path length [cm].

    Returns:
        X: Species mole fractions `(N, ...)`.

    """
    A = np.asarray(A)
    sigma = np.asarray(sigma)
    T = np.asarray(T)
    P = np.asarray(P)

    if A.ndim == 1:
        assert sigma.ndim == 2 and T.shape == (1,) and P.shape == (1,)
    elif A.ndim == 2:
        # For absorbance time histories, the arrays must be reshaped so that time is the outer
        # axis, as the NumPy linear algebra routines operate on the inner matrices

        A = np.moveaxis(A, -1, 0)
        if sigma.ndim == 3:
            sigma = np.moveaxis(sigma, -1, 0)
        else:
            sigma = np.broadcast_to(sigma, (A.shape[0],) + sigma.shape)

    X = np.linalg.solve(sigma / 1e6 * AVOGADRO_NUMBER * P / (GAS_CONSTANT * T) * L, A)

    # For absorbance time histories, reshape the array so that time is the inner axis
    if X.ndim == 2:
        np.moveaxis(X, 0, 1)

    return X
