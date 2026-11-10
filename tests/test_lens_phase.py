"""calculate_phix_phiy: sign convention of the pi/2 term (no external data needed)."""
import numpy as np
import pytest

from Astimator import calculate_phix_phiy, tilted_lens_focal_lengths


def _original(fx, fy, waistx, waisty, lamb, z0, z, flip_lens=False):
    """The original formula (+pi/2), copied verbatim as reference."""
    zrx = np.pi * (waistx) ** 2 / lamb
    zry = np.pi * (waisty) ** 2 / lamb
    if flip_lens:
        fx, fy = fy, fx
    zbarx = zrx * (1 + (z0 / zrx) ** 2)
    zbary = zry * (1 + (z0 / zry) ** 2)
    Rx = (z0 ** 2 + zrx ** 2) / z0
    Ry = (z0 ** 2 + zry ** 2) / z0
    phix = np.pi / 2 - np.arctan2(zbarx * (fx * Rx - z * Rx + z * fx), z * fx * Rx) + np.arctan(z0 / zrx)
    phiy = np.pi / 2 - np.arctan2(zbary * (fy * Ry - z * Ry + z * fy), z * fy * Ry) + np.arctan(z0 / zry)
    return phix, phiy


ARGS = dict(waistx=700e-6, waisty=700e-6, lamb=633e-9, z0=1.18)


@pytest.mark.parametrize("theta,z", [(0, 0.30), (20, 0.22), (35, 0.26)])
@pytest.mark.parametrize("flip_lens", [False, True])
def test_flip_coordinates_true_matches_original(theta, z, flip_lens):
    fx, fy = tilted_lens_focal_lengths("biconvex", theta, f0=0.3, n=1.5151)
    got = calculate_phix_phiy(fx, fy, z=z, flip_lens=flip_lens, flip_coordinates=True, **ARGS)
    ref = _original(fx, fy, z=z, flip_lens=flip_lens, **ARGS)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)


@pytest.mark.parametrize("theta,z", [(0, 0.30), (20, 0.22)])
def test_default_is_physical_minus_pi_over_2(theta, z):
    fx, fy = tilted_lens_focal_lengths("biconvex", theta, f0=0.3, n=1.5151)
    phys = calculate_phix_phiy(fx, fy, z=z, **ARGS)
    flip = calculate_phix_phiy(fx, fy, z=z, flip_coordinates=True, **ARGS)
    np.testing.assert_allclose(np.subtract(flip, phys), [np.pi, np.pi], atol=1e-14)
    ref = _original(fx, fy, z=z, **ARGS)
    np.testing.assert_allclose(phys, np.subtract(ref, np.pi), atol=1e-14)
