# hg_estimator

Estimate Hermite-Gauss coefficients from a set of intensity images taken at different
lens tilts / distances.

```python
import numpy as np
from hg_estimator import CoefficientEstimator, compute_phases

# basis from structured_optics (same pixel grid as the images)
basis = hg_basis(beam, N, norm1=True)

est = CoefficientEstimator(basis, N)           # basis is processed once, reuse it

phix, phiy = zip(*[
    compute_phases("biconvex", th, z, waistx=700e-6, waisty=700e-6,
                   lamb=633e-9, z0=1.18, f0=0.30, n=1.5151)
    for z, th in [(0.30, 0), (0.22, 20)]
])

result = est.estimate([img1, img2], phix, phiy, rng=0)
result.coefs      # (N+1, N+1) complex, unit norm, C[0,0] real
result.loss       # final negative log-likelihood
```

Repetitions / several basis waists are up to the caller, e.g.:

```python
best = min((est.estimate(imgs, phix, phiy, rng=s) for s in range(5)), key=lambda r: r.loss)
```

Nothing is written to disk; saving is up to you.
