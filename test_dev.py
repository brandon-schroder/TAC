from elements import InletBoundary, Expander, BladeRow, Reducer
from subelements import LossModel, DeviationModel, BlockageModel
from flowstation import link_ports
import numpy as np

N = 10  # streamlines

loss_r  = [LossModel("L", 0.05) for _ in range(N)]
dev_r   = [DeviationModel("D", delta_0=np.radians(2)) for _ in range(N)]
blk_r   = [BlockageModel("B", blockage_fraction=0.02) for _ in range(N)]

inlet  = InletBoundary("inlet", m=10.0, pt=101325., tt=288., r=0.15,
                        v_theta=0., v_m=150.)
exp    = Expander("exp", N, r_hub=0.10, r_tip=0.20)
rotor  = BladeRow("rotor", N, r_hub=0.10, r_tip=0.20,
                  beta_le_dist=[np.radians(50)]*N,
                  beta_te_dist=[np.radians(-30)]*N,
                  loss_models=loss_r, deviation_models=dev_r,
                  blockage_models=blk_r)
rdcr   = Reducer("rdcr", N)

# Link ports
inlet.execute()
link_ports(inlet.Fl_O, exp.Fl_I)
exp.execute()

# Connect Expander outlets to BladeRow inlet (aggregate)
link_ports(exp.Fl_O, rotor.Fl_I)
# Also push per-stream states into each StreamSegment
for i, seg in enumerate(rotor.segments):
    link_ports(exp.stream_outlets[i], seg.Fl_I)

rotor.execute(omega=1000.)  # rad/s

# Collect StreamSegment exits into Reducer
for i, seg in enumerate(rotor.segments):
    link_ports(seg.Fl_O, rdcr.stream_inlets[i])
rdcr.execute()
rdcr.Fl_O.report()