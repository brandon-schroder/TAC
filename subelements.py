import numpy as np
from thermo import CP, GAMMA, R


class LossModel:
    """
    Total pressure loss subelement.

    Reads rho and w directly from the inlet FlowStation — no internal
    thermodynamic recomputation. Works for scalar or array inlet state.

    Loss coefficient definition:
        omega_bar = delta_Pt / q_rel,  q_rel = 0.5 * rho * W^2
    """

    def __init__(self, name: str, loss_coefficient: float):
        self.name             = name
        self.loss_coefficient = loss_coefficient

    def compute(self, inlet) -> np.ndarray:
        """
        Returns delta_Pt [Pa]. Scalar or array depending on inlet.

        Requires inlet._resolve_state(beta) to have been called so
        inlet.rho and inlet.w are populated with relative-frame values.
        """
        q_rel = 0.5 * inlet.rho * inlet.w**2
        return self.loss_coefficient * q_rel


class DeviationModel:
    """
    Exit flow deviation subelement.

    Reads inlet.beta and exit_fs.ma_r directly from FlowStation objects.
    Works for scalar or array state.

    Model:
        delta = delta_0 + k_inc * incidence + k_mach * Ma_rel_exit
    """

    def __init__(self, name: str, delta_0: float,
                 k_inc: float = 0.10, k_mach: float = 0.05):
        self.name    = name
        self.delta_0 = np.asarray(delta_0, dtype=float)
        self.k_inc   = k_inc
        self.k_mach  = k_mach

    def compute(self, inlet, beta_le, exit_fs) -> np.ndarray:
        """
        Returns deviation angle delta [rad]. Scalar or array.

        inlet    -- inlet FlowStation (relative frame resolved)
        beta_le  -- blade leading-edge angle [rad], scalar or array
        exit_fs  -- exit FlowStation (trial state, relative frame resolved)
        """
        incidence = inlet.beta - np.asarray(beta_le, dtype=float)
        return self.delta_0 + self.k_inc * incidence + self.k_mach * exit_fs.ma_r