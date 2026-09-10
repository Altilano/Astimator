import numpy as np


def compute_transform_phases(transforms, ni, mi, waist, lamb, z0, f):
    """
    transforms: list[TransformSpec]
    ni, mi:     HG index grids returned by beam's hg_projector/hg_basis
    returns:    dict {transform.name: raveled complex phase array exp(i*(ax*ni+ay*mi))}

    Requires `calculate_ax_ay` to be importable (from your existing utils.py,
    same signature as in both original notebooks).
    """
    from utils import calculate_ax_ay  # your existing optics function

    phases = {}
    for t in transforms:
        ax, ay = calculate_ax_ay(
            waist=waist,
            lamb=lamb,
            z0=z0,
            f=f,
            theta=t.theta,
            z=t.z,
            flip_lens=t.flip_lens,
        )
        phases[t.name] = np.exp(1j * (ax * ni + ay * mi)).ravel()
    return phases
