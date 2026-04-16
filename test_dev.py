from dev import *


# =============================================================================
# 1. MAIN
# =============================================================================
if __name__ == "__main__":

    # Subelements constructed independently and injected into BladeRow
    rotor_loss  = LossModel("Rotor1.Loss",  loss_coefficient=0.05)
    stator_loss = LossModel("Stator1.Loss", loss_coefficient=0.03)

    rotor_dev   = DeviationModel("Rotor1.Dev",  delta_0=np.radians(2.0),
                                 k_inc=0.10, k_mach=0.05)
    stator_dev  = DeviationModel("Stator1.Dev", delta_0=np.radians(1.0),
                                 k_inc=0.08, k_mach=0.03)

    # Elements
    start = InletBoundary(
        "Start", m=10.0, pt=101325.0, tt=288.15,
        r=0.3, v_theta=0.0, v_m=85.0
    )

    rotor_1 = BladeRow(
        name="Rotor1", r_exit=0.3, a_exit=0.1, a_blockage=0.005,
        beta_le=np.radians(50.0), beta_te=np.radians(40.0),
        loss_model=rotor_loss, deviation_model=rotor_dev,
        mode="des"
    )

    stator_1 = BladeRow(
        name="Stator1", r_exit=0.3, a_exit=0.1, a_blockage=0.005,
        beta_le=np.radians(-20.0), beta_te=np.radians(-5.0),
        loss_model=stator_loss, deviation_model=stator_dev,
        mode="des"
    )

    # Assemble
    system = TurboSystem("SingleStage")
    system.add_element(start)
    system.add_element(rotor_1)
    system.add_element(stator_1)

    system.connect(start.Fl_O,   rotor_1.Fl_I)
    system.connect(rotor_1.Fl_O, stator_1.Fl_I)

    omega_map = {
        "Start":   0.0,
        "Rotor1":  800.0,
        "Stator1": 0.0,
    }

    system.run(omega_map=omega_map, n_iter=15, tol=1.0)
    system.report()