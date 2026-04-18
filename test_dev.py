import numpy as np
from elements import InletBoundary, BladeRow
from subelements import LossModel, DeviationModel
from flowstation import link_ports

# =============================================================================
# Single-stage axial compressor — meanline
# =============================================================================
R_MEAN = 0.15    # m
OMEGA  = 1000.0  # rad/s

# Blade angles (relative frame, compressor convention: positive = forward sweep)
BETA_LE = np.radians(50.0)
BETA_TE = np.radians(-30.0)

# Flow area at rotor exit (will be recomputed in design mode)
A_EXIT    = np.pi * R_MEAN**2 * 0.3   # initial estimate
A_BLOCK   = 0.02 * A_EXIT

# Subelements
loss = LossModel("rotor_loss", loss_coefficient=0.05)
dev  = DeviationModel("rotor_dev", delta_0=np.radians(2.0),
                      k_inc=0.10, k_mach=0.05)

# Elements
inlet = InletBoundary("inlet", m=10.0, p_t=101325.0, t_t=288.0,
                      r=R_MEAN, v_u=0.0, v_m=150.0)

rotor = BladeRow("rotor", r_exit=R_MEAN, a_exit=A_EXIT, a_blockage=A_BLOCK,
                 beta_le=BETA_LE, beta_te=BETA_TE,
                 loss_model=loss, deviation_model=dev, mode="des")

# --- Execute ---
inlet.execute()
link_ports(inlet.Fl_O, rotor.Fl_I)
rotor.execute(omega=OMEGA)

# --- Report ---
print("\n=== MEANLINE RESULTS ===")
inlet.Fl_O.report()
rotor.Fl_O.report()
rotor.report()

# Sanity checks
assert rotor.Fl_O.p_t > inlet.Fl_O.p_t,   "Compressor must raise total pressure"
assert rotor.Fl_O.h_t > inlet.Fl_O.h_t,   "Compressor must add enthalpy"
assert rotor.Fl_O.v_m > 0,                 "Meridional velocity must be positive"
print("\nAll sanity checks passed.")