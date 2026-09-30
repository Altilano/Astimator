"""Helpers that compute the per-image Gouy-like phases (phi_x, phi_y)."""
import math
import numpy as np


def tilted_lens_focal_lengths(lens_type, theta_deg, f0=None, R=None, n=None):
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


def calculate_phix_phiy(fx, fy, waistx, waisty, lamb, z0, z, flip_lens=False):
    zrx = np.pi * (waistx) ** 2 / lamb
    zry = np.pi * (waisty) ** 2 / lamb
    if flip_lens:
        fx, fy = fy, fx
    zbarx = zrx * (1 + (z0 / zrx) ** 2)
    zbary = zry * (1 + (z0 / zry) ** 2)
    Rx = (z0**2 + zrx**2) / z0
    Ry = (z0**2 + zry**2) / z0
    phix = np.pi / 2 - np.arctan2(zbarx * (fx * Rx - z * Rx + z * fx), z * fx * Rx) + np.arctan(z0 / zrx)
    phiy = np.pi / 2 - np.arctan2(zbary * (fy * Ry - z * Ry + z * fy), z * fy * Ry) + np.arctan(z0 / zry)
    return phix, phiy


def compute_phases(lens_type, theta_deg, z, *, waistx, waisty, lamb, z0,
                   f0=None, R=None, n=None, flip_lens=False):
    """Convenience wrapper: lens geometry -> (phi_x, phi_y) for one image.

    Parameters
    ----------
    lens_type : 'biconvex' or 'plano-convex'
    theta_deg : lens tilt angle in degrees
    z         : lens-to-camera distance for this image
    waistx, waisty, lamb, z0 : input-beam parameters
    f0, R, n  : provide exactly two (see ``tilted_lens_focal_lengths``)
    """
    fx, fy = tilted_lens_focal_lengths(lens_type, theta_deg, f0=f0, R=R, n=n)
    return calculate_phix_phiy(fx, fy, waistx, waisty, lamb, z0, z, flip_lens=flip_lens)
