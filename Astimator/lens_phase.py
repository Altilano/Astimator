"""
Lens geometry and the Hermite-Gauss phases (phi_x, phi_y) gained through a
(tilted) lens followed by free propagation.

HG mode (n, m) is multiplied by exp(i * (phi_x * n + phi_y * m)); see
lens_transform.py for how those phases are applied to a basis.

Sign convention of the pi/2 term
--------------------------------
The physical phase is

    phi = arctan(z0 / zr) - arctan2(...) - pi/2

(the default, ``flip_coordinates=False``). If the picture is mirrored with
respect to the basis -- e.g. the beam goes through a 4F system before the
camera, as in the original Jupyter notebooks -- the phase that matches the
data has +pi/2 instead; pass ``flip_coordinates=True`` for that case.
"""
import math

import numpy as np


def tilted_lens_focal_lengths(lens_type, theta_deg, f0=None, R=None, n=None):
    """Effective focal lengths (f_x, f_y) of a lens tilted by theta_deg.

    lens_type: 'biconvex' or 'plano-convex'. Give exactly two of (f0, R, n).
    """
    lens_type = lens_type.strip().lower()
    if lens_type in ("biconvex", "bi-convex", "double-convex"):
        factor = 2.0
    elif lens_type in ("plano-convex", "planoconvex", "plano convex"):
        factor = 1.0
    else:
        raise ValueError("lens_type must be 'biconvex' or 'plano-convex'")

    given = [p is not None for p in (f0, R, n)]
    if sum(given) != 2:
        raise ValueError("Provide exactly two of (f0, R, n).")

    if f0 is None:
        f0 = R / (factor * (n - 1))
    elif R is None:
        R = factor * f0 * (n - 1)
    else:
        n = 1.0 + R / (factor * f0)

    theta = math.radians(theta_deg)
    sin_theta_prime = math.sin(theta) / n
    if abs(sin_theta_prime) > 1.0:
        raise ValueError("No real refraction angle: |sin(theta)/n| > 1")
    theta_prime = math.asin(sin_theta_prime)

    denom = n * math.cos(theta_prime) - math.cos(theta)
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("focal length diverges at this angle")

    f_x = f0 * (n - 1) * math.cos(theta) ** 2 / denom
    f_y = f0 * (n - 1) / denom
    return f_x, f_y


def calculate_phix_phiy(fx, fy, waistx, waisty, lamb, z0, z,
                        flip_lens=False, flip_coordinates=False):
    """HG phases (phi_x, phi_y) per mode index, for a lens at distance z0 from
    the waist and a picture taken a distance z after the lens.

    flip_lens:        swap fx and fy (lens rotated by 90 degrees).
    flip_coordinates: False (default) -> physical phase, with -pi/2.
                      True -> the phase with +pi/2, valid when the picture is
                      mirrored w.r.t. the basis (e.g. after a 4F system).
    """
    zrx = np.pi * (waistx) ** 2 / lamb
    zry = np.pi * (waisty) ** 2 / lamb
    if flip_lens:
        fx, fy = fy, fx
    zbarx = zrx * (1 + (z0 / zrx) ** 2)
    zbary = zry * (1 + (z0 / zry) ** 2)
    Rx = (z0 ** 2 + zrx ** 2) / z0
    Ry = (z0 ** 2 + zry ** 2) / z0
    offset = np.pi / 2 if flip_coordinates else -np.pi / 2
    phix = offset - np.arctan2(zbarx * (fx * Rx - z * Rx + z * fx), z * fx * Rx) + np.arctan(z0 / zrx)
    phiy = offset - np.arctan2(zbary * (fy * Ry - z * Ry + z * fy), z * fy * Ry) + np.arctan(z0 / zry)
    return phix, phiy
