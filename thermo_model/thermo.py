import numpy as np
import cantera as ct
from numba import float64
from numba.experimental import jitclass
import time

# ==========================================
# 1. THE EVALUATOR (Compiled via Numba)
# ==========================================

# Define the C-types for the Numba class properties
spec = [
    ('T_pts', float64[:]),
    ('P_pts', float64[:]),
    ('props_grid', float64[:, :, :]),  # [T_idx, P_idx, property_idx]
    ('h_pts', float64[:]),  # X-axis for h-P reverse lookups
    ('s_pts', float64[:]),  # X-axis for s-P reverse lookups
    ('T_grid_rev_hP', float64[:, :]),  # Y-axis is always P_pts
    ('h_grid_rev_sP', float64[:, :]),
    ('T_grid_rev_sP', float64[:, :])
]


@jitclass(spec)
class FastThermoEvaluator:
    def __init__(self, T_pts, P_pts, props_grid, h_pts, s_pts,
                 T_grid_rev_hP, h_grid_rev_sP, T_grid_rev_sP):
        """A compiled evaluator executing bilinear interpolations at machine speed."""
        self.T_pts = T_pts
        self.P_pts = P_pts
        self.props_grid = props_grid
        self.h_pts = h_pts
        self.s_pts = s_pts

        self.T_grid_rev_hP = T_grid_rev_hP
        self.h_grid_rev_sP = h_grid_rev_sP
        self.T_grid_rev_sP = T_grid_rev_sP

    def _get_indices_weights(self, x, y, X, Y):
        """Finds grid indices and interpolation weights, allowing linear extrapolation."""
        # Find X bounding indices
        i = np.searchsorted(X, x) - 1
        if i < 0:
            i = 0
        elif i >= len(X) - 1:
            i = len(X) - 2

        # Find Y bounding indices
        j = np.searchsorted(Y, y) - 1
        if j < 0:
            j = 0
        elif j >= len(Y) - 1:
            j = len(Y) - 2

        # Calculate geometric weights (0.0 to 1.0)
        wx = (x - X[i]) / (X[i + 1] - X[i])
        wy = (y - Y[j]) / (Y[j + 1] - Y[j])

        return i, j, wx, wy

    def _interp_2d(self, x, y, X, Y, Z_grid):
        """Generic, highly-optimized bilinear interpolator."""
        i, j, wx, wy = self._get_indices_weights(x, y, X, Y)

        z11 = Z_grid[i, j]
        z12 = Z_grid[i, j + 1]
        z21 = Z_grid[i + 1, j]
        z22 = Z_grid[i + 1, j + 1]

        z_y1 = z11 + wx * (z21 - z11)
        z_y2 = z12 + wx * (z22 - z12)
        return z_y1 + wy * (z_y2 - z_y1)

    # --- Public API Methods (Explicit & Fast) ---

    def state_from_TP(self, T, P):
        """Returns array: [h, s, rho, cp, gamma] given T(K) and P(Pa)"""
        i, j, wx, wy = self._get_indices_weights(T, P, self.T_pts, self.P_pts)
        res = np.zeros(5, dtype=np.float64)

        for k in range(5):
            z11 = self.props_grid[i, j, k]
            z12 = self.props_grid[i, j + 1, k]
            z21 = self.props_grid[i + 1, j, k]
            z22 = self.props_grid[i + 1, j + 1, k]

            z_y1 = z11 + wx * (z21 - z11)
            z_y2 = z12 + wx * (z22 - z12)
            res[k] = z_y1 + wy * (z_y2 - z_y1)

        return res

    def T_from_hP(self, h, P):
        """Returns actual Temperature (K) given actual Enthalpy and Pressure"""
        return self._interp_2d(h, P, self.h_pts, self.P_pts, self.T_grid_rev_hP)

    def h_from_sP(self, s, P):
        """Returns ideal Enthalpy (J/kg) given inlet Entropy and exit Pressure"""
        return self._interp_2d(s, P, self.s_pts, self.P_pts, self.h_grid_rev_sP)

    def T_from_sP(self, s, P):
        """Returns ideal Temperature (K) given inlet Entropy and exit Pressure"""
        return self._interp_2d(s, P, self.s_pts, self.P_pts, self.T_grid_rev_sP)


# ==========================================
# 2. THE BUILDER (Pure Python/Cantera)
# ==========================================

class ThermoTableBuilder:

    @staticmethod
    def _create_reverse_grid(P_pts, primary_grid, target_grid, grid_size):
        """Generic builder to invert axes for reverse lookups."""
        p_min, p_max = np.min(primary_grid), np.max(primary_grid)
        pts_rev = np.linspace(p_min, p_max, grid_size[0])
        grid_rev = np.zeros(grid_size, dtype=np.float64)

        for j, P in enumerate(P_pts):
            primary_slice = primary_grid[:, j]
            target_slice = target_grid[:, j]
            grid_rev[:, j] = np.interp(pts_rev, primary_slice, target_slice)

        return pts_rev, grid_rev

    @classmethod
    def build(cls, fluid_name='air.yaml', T_range=(200, 2500), P_range=(1e4, 5e6), grid_size=(200, 200)):
        """Builds the grids using Cantera and returns a compiled FastThermoEvaluator."""
        print(f"Building base property arrays for {fluid_name} using Cantera...")
        start_time = time.time()

        fluid = ct.Solution(fluid_name)
        T_pts = np.linspace(T_range[0], T_range[1], grid_size[0])
        P_pts = np.linspace(P_range[0], P_range[1], grid_size[1])

        # Order: [h, s, rho, cp, gamma]
        props_grid = np.zeros((grid_size[0], grid_size[1], 5), dtype=np.float64)
        T_grid = np.zeros(grid_size, dtype=np.float64)

        for i, T in enumerate(T_pts):
            for j, P in enumerate(P_pts):
                fluid.TP = T, P
                T_grid[i, j] = T
                props_grid[i, j, 0] = fluid.enthalpy_mass
                props_grid[i, j, 1] = fluid.entropy_mass
                props_grid[i, j, 2] = fluid.density
                props_grid[i, j, 3] = fluid.cp_mass
                props_grid[i, j, 4] = fluid.cp_mass / fluid.cv_mass

        print("Inverting arrays for reverse lookups...")
        h_grid = props_grid[:, :, 0]
        s_grid = props_grid[:, :, 1]

        # Build specific reverse grids using the generic helper
        h_pts, T_grid_rev_hP = cls._create_reverse_grid(P_pts, h_grid, T_grid, grid_size)
        s_pts, h_grid_rev_sP = cls._create_reverse_grid(P_pts, s_grid, h_grid, grid_size)
        _, T_grid_rev_sP = cls._create_reverse_grid(P_pts, s_grid, T_grid, grid_size)

        build_time = time.time() - start_time
        print(f"Table generation complete in {build_time:.2f} seconds.")
        print("Compiling Numba engine...")

        return FastThermoEvaluator(T_pts, P_pts, props_grid, h_pts, s_pts,
                                   T_grid_rev_hP, h_grid_rev_sP, T_grid_rev_sP)


