"""Carry the uncertainty of fitted coefficients through to a predicted output.

A fit reports its coefficients, their covariance and, for a cluster bootstrap, the
coefficient sets it resampled. This module draws coefficient sets from that description and
applies any function of them, so the spread of the output describes the uncertainty of the fit's
coefficients; the template, the assay's enzyme amount
and the rest of the definition are taken as exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

MAX_ATTEMPT_FACTOR = 50


def covariance_of(fit: dict[str, Any]) -> tuple[list[str], Any]:
    """The coefficient names of a stored fit and their covariance, in the fit's units."""
    import numpy as np

    dependence = fit.get("parameter_dependence") or {}
    covariance = dependence.get("covariance")
    if covariance is None:
        raise ValueError(
            "The fit has no parameter covariance, so its uncertainty cannot be "
            "propagated. A fit has no covariance when a coefficient is at a bound, the "
            "local Jacobian is rank-deficient or ill-conditioned, or a bootstrap "
            "resample failed."
        )
    return list(dependence["order"]), np.asarray(covariance, dtype=float)


def draw_normal(
    fit: dict[str, Any], samples: int, seed: int, positive: Sequence[str] = ()
) -> tuple[list[str], Any, int]:
    """Coefficient sets from the normal approximation, with a count of those rejected.

    A coefficient listed in `positive` cannot be negative, so draws with a negative one
    are discarded and redrawn; the number discarded is returned, because a large number
    means the normal approximation is a poor description of this fit.
    """
    import numpy as np

    names, covariance = covariance_of(fit)
    values = {p["name"]: float(p["value"]) for p in fit["parameters"]}
    mean = np.array([values[n] for n in names])
    rng = np.random.default_rng(seed)
    keep = [i for i, n in enumerate(names) if n in positive]
    accepted: list[Any] = []
    rejected = 0
    attempts = 0
    while len(accepted) < samples:
        attempts += 1
        if attempts > MAX_ATTEMPT_FACTOR:
            raise ValueError(
                "More than 98 % of the draws had a negative coefficient. The normal "
                "approximation does not describe this fit; use the cluster bootstrap."
            )
        batch = rng.multivariate_normal(mean, covariance, size=samples, method="eigh")
        good = np.all(batch[:, keep] > 0, axis=1) if keep else np.ones(len(batch), bool)
        rejected += int((~good).sum())
        accepted.extend(batch[good].tolist())
    return names, np.asarray(accepted[:samples]), rejected


def draw_bootstrap(
    fit: dict[str, Any], samples: int, seed: int
) -> tuple[list[str], Any]:
    """Coefficient sets resampled, with replacement, from the fit's bootstrap draws."""
    import numpy as np

    draws = (fit.get("uncertainty") or {}).get("parameter_draws")
    if not draws:
        raise ValueError(
            "The fit contains no bootstrap draws. Fit with uncertainty cluster_bootstrap, "
            "or set distribution to normal."
        )
    names = [p["name"] for p in fit["parameters"]]
    array = np.asarray(draws, dtype=float)
    rng = np.random.default_rng(seed)
    return names, array[rng.integers(0, len(array), size=samples)]


def quantiles_of(values: Sequence[float], levels: Sequence[float]) -> dict[str, float]:
    import numpy as np

    found = np.quantile(np.asarray(values, dtype=float), list(levels))
    return {f"{level:g}": float(v) for level, v in zip(levels, found, strict=True)}


def propagate(
    draws: Any, function: Callable[[Sequence[float]], float], levels: Sequence[float]
) -> dict[str, Any]:
    """The mean, spread and quantiles of `function` over coefficient sets."""
    import numpy as np

    outputs = np.array([function(row) for row in draws], dtype=float)
    return {
        "mean": float(outputs.mean()),
        "standard_deviation": float(outputs.std(ddof=1)) if len(outputs) > 1 else 0.0,
        "quantiles": quantiles_of(outputs, levels),
        "values": outputs.tolist(),
    }


def linear_interval(
    mean: Sequence[float], covariance: Any, gradient: Sequence[float], level: float
) -> tuple[float, float, float]:
    """The exact interval of a function linear in the coefficients, for normal draws.

    Returns the value at the mean and the lower and upper bounds enclosing `level` of
    the distribution: value -/+ z sqrt(g' C g).
    """
    import numpy as np
    from scipy.stats import norm

    g = np.asarray(gradient, dtype=float)
    centre = float(g @ np.asarray(mean, dtype=float))
    sd = math.sqrt(max(float(g @ np.asarray(covariance, dtype=float) @ g), 0.0))
    z = float(norm.ppf(0.5 + level / 2.0))
    return centre, centre - z * sd, centre + z * sd
