from scipy.optimize import root_scalar


GAS_CONSTANT = 8.31432e3
"Universal gas constant [J mol^-1 K^-1]"


def solve_driver_pressure(
    P2: float,
    U2: float,
    MW4: float,
    gamma4: float,
    T4: float,
    area_ratio: float = 1,
):
    """

    Args:
        P2: Pressure behind incident shock [Pa].
        U2: Velocity of gas behind incident shock (lab-frame coordinates) [m/s].
        MW4: Mean molecular weight of driver gas [g/mol].
        gamma4: Specific heat ratio of driver gas.
        T4: Initial temperature of driver gas [K].
        area_ratio:

    Returns:

    """

    if area_ratio < 1:
        raise ValueError("Area ratio must be greater than or equal to one.")

    a4 = (gamma4 * GAS_CONSTANT / MW4 * T4) ** 0.5

    def equivalence_factor(_M3a, _Me):
        return (((2 + (gamma4 - 1) * _M3a ** 2) / (
                2 + (gamma4 - 1) * _Me ** 2)) ** 0.5
                * (2 + (gamma4 - 1) * _Me) / (2 + (gamma4 - 1) * _M3a)) ** (
                       2 * gamma4 / (gamma4 - 1))

    def calc_M3(_M3a, _Me):
        return 1 / (a4 / U2 * equivalence_factor(_M3a, _Me) ** (
                (gamma4 - 1) / gamma4 / 2) - (gamma4 - 1) / 2)

    def solve_M3a(_Me):
        def area_ratio_error(_M3a):
            return area_ratio * _M3a - _Me * ((2 + (gamma4 - 1) * _M3a ** 2) / (
                    2 + (gamma4 - 1) * _Me ** 2)) \
                   ** ((gamma4 + 1) / (gamma4 - 1) / 2)

        _r = root_scalar(area_ratio_error, bracket=[0, 1])
        if not _r.converged:
            raise RuntimeError("Root finding routine for M3a did not converge")
        return _r.root

    # Assume supersonic case (Me = 1)
    Me = 1
    M3a = solve_M3a(Me)
    M3 = calc_M3(M3a, Me)

    # Subsonic case (M3 = Me)
    if M3 < 1:
        r = root_scalar(
            lambda _M3: _M3 - calc_M3(solve_M3a(_Me=_M3), _Me=_M3), bracket=[0, 1])
        if not r.converged:
            raise RuntimeError("Root finding routine for M3 did not converge")
        M3 = r.root
        Me = M3
        M3a = solve_M3a(Me)

    return P2 / equivalence_factor(M3a, Me) * (1 + (gamma4 - 1) / 2 * M3) ** (2 * gamma4 / (gamma4 - 1))
