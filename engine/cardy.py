# Cardy's crossing-probability formula for critical 2D percolation, plus the small analysis helpers
# At the critical point the probability that an open cluster spans a rectangle left-to-right depends only on the rectangle's aspect ratio (a consequence of conformal invariance of the scaling limit) 
# This is given exactly by Cardy's formula:
# The crossing probability is a UNIVERSAL FUNCTION of the shape, a much stronger fingerprint of the 2D percolation class than the critical exponents (a whole curve, not two numbers), and it tests conformal invariance directly. 
# We anchor the operational critical point at the p where the SQUARE crosses at 1/2
# We then read the other aspect ratios at that same p and compare to this formula. See runner/cardy_runner.py.

import numpy as np
from scipy.special import ellipk, hyp2f1, gamma
from scipy.optimize import brentq

# Cardy's prefactor: 3 Gamma(2/3) / Gamma(1/3)^2 .
_C = 3.0 * gamma(2.0 / 3.0) / gamma(1.0 / 3.0) ** 2

# Cardy's crossing probability as a function of the cross-ratio eta in (0,1)
def _pi_of_eta(eta):
    return float(_C * eta ** (1.0 / 3.0) * hyp2f1(1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, eta))

# Rectangle aspect ratio H/W as a function of the cross-ratio eta, via the ratio of complete elliptic integrals
def _aspect_HW(eta):
    return float(ellipk(eta) / ellipk(1.0 - eta))

# Exact horizontal (left-right) crossing probability for a critical rectangle of aspect ratio a = W/H.
def cardy_pi_h(aspect):
    a = float(aspect)
    eta = brentq(lambda e: _aspect_HW(e) - 1.0 / a, 1e-12, 1.0 - 1e-12)
    return _pi_of_eta(eta)


# Left-right crossing probability at occupation p_star: the fraction of trials whose left-right
# spanning ONSET density (pR) is <= p_star. pR is exactly what percolationStatsI_par records.
def crossing_probability(pR, p_star):
    return float(np.mean(np.asarray(pR, dtype=float) <= p_star))
