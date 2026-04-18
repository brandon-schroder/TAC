import numpy as np

GAMMA = 1.4
R     = 287.0
CP    = GAMMA * R / (GAMMA - 1.0)

T_REF = 288.15   # entropy reference temperature [K]
P_REF = 101325.0  # entropy reference pressure    [Pa]


def entropy_from_pt_tt(pt, tt):
    """
    Specific entropy [J/kg/K]. Scalar or array.
        s = Cp * ln(Tt/T_ref) - R * ln(Pt/P_ref)
    """
    pt = np.asarray(pt, dtype=float)
    tt = np.asarray(tt, dtype=float)
    return CP * np.log(tt / T_REF) - R * np.log(pt / P_REF)


def pt_from_ht_s(ht, s):
    """
    Total pressure from total enthalpy and entropy. Scalar or array.
        Pt = P_ref * exp((Cp * ln(Tt/T_ref) - s) / R)
    """
    ht = np.asarray(ht, dtype=float)
    s  = np.asarray(s,  dtype=float)
    tt = ht / CP
    return P_REF * np.exp((CP * np.log(tt / T_REF) - s) / R)


def tt_from_ht(ht):
    """Total temperature from total enthalpy. Scalar or array."""
    return np.asarray(ht, dtype=float) / CP