import numpy as np
from numpy.polynomial.hermite import hermval
from math import factorial, pi, sqrt
from hg_estimator import CoefficientEstimator


def hg1d(n, x, w):
    c = np.zeros(n + 1); c[n] = 1
    return hermval(sqrt(2) * x / w, c) * np.exp(-x**2 / w**2) / sqrt(2**n * factorial(n))


def make_basis(N, res=64, w=1.0):
    x = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, x)
    basis = np.empty((N + 1, N + 1), dtype=object)
    for i in range(N + 1):
        for j in range(N + 1):
            b = hg1d(i, X, w) * hg1d(j, Y, w)
            basis[i, j] = b / np.sqrt((np.abs(b) ** 2).sum())
    return basis


def test_recovers_coefs():
    N = 3
    basis = make_basis(N)
    rng = np.random.default_rng(0)
    C = rng.standard_normal((N + 1, N + 1)) + 1j * rng.standard_normal((N + 1, N + 1))
    n, m = np.meshgrid(range(N + 1), range(N + 1), indexing="ij")
    C[n + m > N] = 0
    C /= np.linalg.norm(C)

    phix = [0.0, 0.9, 2.1, 3.0]
    phiy = [0.0, 1.3, 0.4, 2.2]
    imgs = []
    for px, py in zip(phix, phiy):
        E = sum(C[i, j] * np.exp(1j * (px * i + py * j)) * basis[i, j]
                for i in range(N + 1) for j in range(N + 1))
        I = np.abs(E) ** 2
        imgs.append(I / I.max())

    est = CoefficientEstimator(basis, N)
    best = min((est.estimate(imgs, phix, phiy, rng=s) for s in range(5)), key=lambda r: r.loss)
    Cg = C * np.exp(-1j * np.angle(C[0, 0]))
    print("loss", best.loss, "err", np.linalg.norm(best.coefs - Cg))
    assert np.linalg.norm(best.coefs - Cg) < 1e-3
