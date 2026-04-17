import numpy as np
from scipy.interpolate import CubicSpline


class SlopeAndCurvature:
    """
    Fits a cubic spline through each streamline's (z, r) history and
    returns the meridional slope angle phi [rad] and radius of curvature
    rs [m] at every station.

    Call once per outer iteration, after all BladeRow radii have been
    updated, before the next BladeRow solve reads phi/rs.

    Parameters
    ----------
    z_stations : (n_stations,)         axial positions [m]
    r_matrix   : (n_stations, n_str)   streamline centroid radii [m]

    Returns
    -------
    phi : (n_stations, n_str)   slope angle [rad]
    rs  : (n_stations, n_str)   radius of curvature [m]  (1e9 = no curvature)
    """

    def __init__(self, name: str = "SlopeAndCurvature"):
        self.name = name

    def compute(self, z_stations: np.ndarray,
                r_matrix: np.ndarray) -> tuple:
        n_stat, n_str = r_matrix.shape
        phi = np.zeros((n_stat, n_str))
        rs  = np.full((n_stat, n_str), 1e9)

        for j in range(n_str):
            cs   = CubicSpline(z_stations, r_matrix[:, j])
            dr   = cs(z_stations, 1)   # dr/dz
            d2r  = cs(z_stations, 2)   # d²r/dz²

            phi[:, j] = np.arctan(dr)

            denom = (1.0 + dr**2) ** 1.5
            kappa = np.abs(d2r) / np.where(denom > 1e-12, denom, 1e-12)

            rs[:, j] = np.full_like(kappa, 1e9)
            mask = np.abs(kappa) > 1e-10
            rs[mask, j] = 1.0 / kappa[mask]

        return phi, rs