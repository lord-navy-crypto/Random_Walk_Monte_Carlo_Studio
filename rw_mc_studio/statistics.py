from __future__ import annotations
import math
from typing import Tuple
from scipy.stats import beta, t


def mean_ci_t(mean: float, sample_sd: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Student-t CI for a population mean estimated from n independent observations."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")
    if n == 1 or sample_sd == 0:
        return float(mean), float(mean)
    alpha = 1.0 - confidence
    critical = float(t.ppf(1.0 - alpha / 2.0, df=n - 1))
    sem = float(sample_sd) / math.sqrt(n)
    return float(mean - critical * sem), float(mean + critical * sem)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact binomial confidence interval; remains informative when successes is 0 or trials."""
    k, n = int(successes), int(trials)
    if n < 1 or not (0 <= k <= n):
        raise ValueError("Require 0 <= successes <= trials and trials >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high
