from elements import InletBoundary, Expander, BladeRow, Reducer
from subelements import LossModel, DeviationModel, BlockageModel
from slopecurvature import SlopeAndCurvature
from flowstation import link_ports
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
N        = 10      # streamlines
R_HUB    = 0.10   # m
R_TIP    = 0.20   # m
OMEGA    = 1000.0  # rad/s
N_ITER   = 15
TOL      = 1.0     # Pa — convergence on aggregate exit Pt

# Axial stations: inlet plane (z=0) and rotor exit plane (z=0.05 m)
# Add more stations as more blade rows are added.
Z_STATIONS = np.array([0.0, 0.05])

# =============================================================================
# Subelements (one per streamtube)
# =============================================================================
loss_r = [LossModel(f"L_{i}", 0.05)                              for i in range(N)]
dev_r  = [DeviationModel(f"D_{i}", delta_0=np.radians(2))        for i in range(N)]
blk_r  = [BlockageModel(f"B_{i}", blockage_fraction=0.02)        for i in range(N)]

# =============================================================================
# Elements
# =============================================================================
inlet = InletBoundary("inlet", m=10.0, p_t=101325., t_t=288.,
                      r=0.15, v_u=0., v_m=150.)

exp   = Expander("exp", N, r_hub=R_HUB, r_tip=R_TIP)

rotor = BladeRow("rotor", N, r_hub=R_HUB, r_tip=R_TIP,
                 beta_le_dist=[np.radians(50)] * N,
                 beta_te_dist=[np.radians(-30)] * N,
                 loss_models=loss_r, deviation_models=dev_r,
                 blockage_models=blk_r)

rdcr  = Reducer("rdcr", N)

# SlopeAndCurvature utility — updates phi/rs on each segment between iterations
sc = SlopeAndCurvature("SC")

# =============================================================================
# Seed: push boundary conditions forward before iteration 1
# =============================================================================
inlet.execute()
link_ports(inlet.Fl_O, exp.Fl_I)
exp.execute()

# Push per-stream states from Expander into each BladeRow StreamSegment.
# This is done ONCE before the loop; execute() will preserve these inlets
# (it only overwrites a segment inlet if seg.Fl_I.r == 0.0).
for i, seg in enumerate(rotor.segments):
    link_ports(exp.stream_outlets[i], seg.Fl_I)

# Also seed the aggregate inlet port so execute() has a valid agg reference
link_ports(exp.Fl_O, rotor.Fl_I)

# =============================================================================
# Outer iteration loop
# =============================================================================
prev_pt = None

for iteration in range(N_ITER):
    print(f"\n--- Iteration {iteration + 1} ---")

    # --- Update slope and curvature from current streamline radii ---
    # Build r_matrix: shape (n_stations, n_streams)
    # Row 0 = inlet centroids (from Expander), Row 1 = rotor exit centroids
    r_inlet_centroids = np.array([seg.Fl_I.r for seg in rotor.segments])
    r_exit_centroids  = np.array([
        0.5 * (rotor._r_bounds[i] + rotor._r_bounds[i + 1])
        for i in range(N)
    ])
    r_matrix = np.column_stack([r_inlet_centroids, r_exit_centroids]).T
    # r_matrix shape is (2, N) — transpose to (n_stations=2, n_streams=N)

    phi, rs = sc.compute(Z_STATIONS, r_matrix)

    # Write phi and rs into each StreamSegment (exit-plane values, row index 1)
    for i, seg in enumerate(rotor.segments):
        seg.phi = phi[1, i]
        seg.rs  = rs[1, i]

    # --- Solve the blade row ---
    rotor.execute(omega=OMEGA)

    # --- Collect segment exits into Reducer ---
    for i, seg in enumerate(rotor.segments):
        link_ports(seg.Fl_O, rdcr.stream_inlets[i])
    rdcr.execute()

    # --- Convergence check on aggregate exit Pt ---
    pt_now = rdcr.Fl_O.p_t
    if prev_pt is not None:
        delta_pt = abs(pt_now - prev_pt)
        print(f"  ΔPt = {delta_pt:.4f} Pa  (tol = {TOL} Pa)")
        if delta_pt < TOL:
            print(f"\n  ✓ Converged after {iteration + 1} iterations.")
            break
    prev_pt = pt_now

else:
    print(f"\n  ⚠ Reached max iterations ({N_ITER}) without convergence.")

# =============================================================================
# Results
# =============================================================================
print("\n=== FINAL RESULTS ===")
inlet.Fl_O.report()
rdcr.Fl_O.report()

print("\n--- Per-stream exit states ---")
for i, seg in enumerate(rotor.segments):
    print(f"  Stream {i:2d} | r={seg.Fl_O.r:.4f} m | "
          f"Pt={seg.Fl_O.p_t:.1f} Pa | Tt={seg.Fl_O.t_t:.2f} K | "
          f"Vm={seg.Fl_O.v_m:.2f} m/s | beta={np.degrees(seg.Fl_O.beta):.1f}°")