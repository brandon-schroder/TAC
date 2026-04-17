from flowstation import *
from thermo import *
import numpy as np


# =============================================================================
# 4. SUBELEMENTS
# =============================================================================

class LossModel:
    """
    Loss subelement. Computes total pressure loss as a function of the
    *current solver iteration's exit state* — called from inside
    BladeRow.residuals() so the solver Jacobian captures ∂(loss)/∂(state).

    This is the key OTAC implicit coupling pattern: subelements are NOT
    pre-computed externally; they are evaluated at every residual call.

    loss_coefficient : float
        ω̄ = ΔPt / q_rel,  where q_rel = 0.5·ρ1·W1²  (relative dynamic pressure)
    """

    def __init__(self, name: str, loss_coefficient: float):
        self.name = name
        self.loss_coefficient = loss_coefficient

    def compute(self, inlet: FlowStation, omega: float,
                r2: float, v_m2: float, beta2: float) -> float:
        """
        Returns ΔPt [Pa] using the current solver state.
        r2, v_m2, beta2 are the solver's current exit guess — included in the
        signature so that future loss models can depend on exit conditions
        (e.g. shock loss, exit Mach).
        """
        u1 = omega * inlet.r
        w_u1 = u1 - inlet.v_u
        w_sq1 = inlet.v_m ** 2 + w_u1 ** 2

        v_sq1 = inlet.v_m ** 2 + inlet.v_u ** 2
        h1 = inlet.h_t - 0.5 * v_sq1
        t1 = h1 / CP
        p1 = inlet.p_t * (t1 / inlet.t_t) ** (GAMMA / (GAMMA - 1.0))
        rho1 = p1 / (R * t1)

        q_rel = 0.5 * rho1 * w_sq1
        return self.loss_coefficient * q_rel


class DeviationModel:
    """
    Deviation subelement. Computes exit flow deviation δ [rad] from the
    current solver state — called from inside BladeRow.residuals().

    Simple empirical model:
        δ = δ0 + k_inc · i + k_mach · M_rel_exit

    Replace the body of compute() with Lieblein or NACA correlations
    without changing the interface.

    Parameters
    ----------
    delta_0 : float  Minimum-loss deviation angle [rad]
    k_inc   : float  Incidence sensitivity [rad/rad]
    k_mach  : float  Exit relative Mach sensitivity [rad]
    """

    def __init__(self, name: str, delta_0: float,
                 k_inc: float = 0.10, k_mach: float = 0.05):
        self.name = name
        self.delta_0 = delta_0
        self.k_inc = k_inc
        self.k_mach = k_mach

    def compute(self, inlet: FlowStation, beta_le: float, omega: float,
                r2: float, v_m2: float, beta2: float) -> float:
        """
        Returns δ [rad].

        inlet, beta_le  — fixed for this element's solve (upstream state)
        r2, v_m2, beta2 — current solver iteration's exit state
        """
        # Incidence (computed from fixed inlet — does not vary with solver vars)
        u1 = omega * inlet.r
        w_u1 = u1 - inlet.v_u
        beta1 = np.arctan2(w_u1, inlet.v_m)
        incidence = beta1 - beta_le

        # Relative exit Mach from current solver state
        u2 = omega * r2
        w_u2 = v_m2 * np.tan(beta2)  # W_u in relative frame
        v_u2 = u2 - w_u2
        v_sq2 = v_m2 ** 2 + v_u2 ** 2
        w_sq2 = v_m2 ** 2 + w_u2 ** 2

        # Approximate static T2 (inlet.h_t used as proxy — solver iterates to convergence)
        h2_approx = inlet.h_t - 0.5 * v_sq2
        t2_approx = max(h2_approx / CP, 1.0)  # guard against negative T
        m_rel2 = np.sqrt(w_sq2 / (GAMMA * R * t2_approx))

        return self.delta_0 + self.k_inc * incidence + self.k_mach * m_rel2


class BlockageModel:
    """
    Wake blockage subelement.  Returns effective area reduction [m²].

    The simple constant-fraction model here is a placeholder — replace
    with a displacement-thickness-based correlation (e.g. Koch-Smith)
    for higher fidelity.  The interface is intentionally identical to
    LossModel and DeviationModel: called inside residuals(), implicit.
    """

    def __init__(self, name: str, blockage_fraction: float = 0.0):
        self.name = name
        self.blockage_fraction = blockage_fraction

    def compute(self, inlet: "FlowStation",
                r_inner: float, r_outer: float) -> float:
        """
        Returns blockage area [m²].

        r_inner, r_outer — current boundary radii for this streamtube,
        updated by BladeRow._update_segment_geometry() before each
        residual evaluation so this call is always seeing current geometry.

        To implement a tip-clearance model that depends on local radius,
        add r_tip as a constructor argument and compare r_outer to r_tip here.
        """
        gross_area = np.pi * (r_outer ** 2 - r_inner ** 2)
        return self.blockage_fraction * gross_area