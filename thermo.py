import numpy as np
# =============================================================================
# 1. GLOBAL THERMO  (perfect gas — real-gas CEA lookup is a future upgrade)
# =============================================================================
GAMMA = 1.4
R     = 287.0
CP    = GAMMA * R / (GAMMA - 1.0)

# Reference entropy anchor: s=0 at (T_REF, P_REF)
T_REF = 288.15
P_REF = 101325.0


def entropy_from_pt_tt(pt: float, tt: float) -> float:
    """
    Specific entropy for a calorically perfect gas, anchored at (T_REF, P_REF).
        s = Cp * ln(Tt/T_ref) - R * ln(Pt/P_ref)
    """
    return CP * np.log(tt / T_REF) - R * np.log(pt / P_REF)


def pt_from_ht_s(ht: float, s: float) -> float:
    """Recover Pt given total enthalpy ht (= Cp*Tt) and entropy s."""
    tt = ht / CP
    return P_REF * np.exp((CP * np.log(tt / T_REF) - s) / R)


def tt_from_ht(ht: float) -> float:
    return ht / CP