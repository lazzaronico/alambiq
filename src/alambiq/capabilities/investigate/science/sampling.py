"""Sampling diagnostics for scalar quantities from independent replicas.

Pure NumPy. The quantities are arranged as an array of replicas by frames by
quantities, with equal numbers of frames per replica.
"""

from __future__ import annotations

from typing import Any


def effective_samples(series):
    """Geyer initial-positive, monotone paired autocorrelation sum, capped at N."""
    import numpy as np

    n = len(series)
    if np.ptp(series) == 0:
        return None
    centered = series - series.mean()
    transformed = np.fft.rfft(centered, n=1 << (2 * n - 1).bit_length())
    acov = np.fft.irfft(transformed * transformed.conjugate())[:n]
    rho = acov / acov[0]
    pairs = rho[: n - n % 2].reshape(-1, 2).sum(axis=1)
    positive = []
    for pair in pairs:
        if pair <= 0:
            break
        positive.append(min(float(pair), positive[-1] if positive else float(pair)))
    tau = max(1.0, -1.0 + 2.0 * sum(positive))
    return float(n / tau)


def sampling_statistics(values, block_counts):
    """Equal-length independent replicas; preserve each uncertainty estimator."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    if (
        x.ndim != 3
        or x.shape[0] < 2
        or x.shape[1] < 40
        or x.shape[2] < 1
        or not np.isfinite(x).all()
        or not isinstance(block_counts, list)
        or not block_counts
        or len(set(block_counts)) != len(block_counts)
        or any(
            type(b) is not int or b < 2 or x.shape[1] % b or x.shape[1] // b < 2
            for b in block_counts
        )
        or x.shape[1] % 20
    ):
        raise ValueError(
            "Use at least 2 replicas of equal length, at least 40 frames divisible by 20, and distinct block counts that each leave at least 2 frames per block"
        )
    m, n, p = x.shape
    means = x.mean(axis=1)
    split = x.reshape(m * 2, n // 2, p)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = (n // 2) * split.mean(axis=1).var(axis=0, ddof=1)
    rhat, ess, diagnostics = [], [], []
    for i in range(p):
        effective = [effective_samples(row[:, i]) for row in x]
        if within[i] == 0 or any(e is None for e in effective):
            rhat.append(None)
            ess.append(None)
            diagnostics.append(
                {
                    "status": "constant"
                    if np.ptp(x[:, :, i]) == 0
                    else "zero-within-chain-variance",
                    "replicaeffective_samples": effective,
                }
            )
        else:
            variance = (1 - 2 / n) * within[i] + between[i] * 2 / n
            rhat.append(float(max(1.0, np.sqrt(variance / within[i]))))
            ess.append(float(sum(effective)))
            diagnostics.append(
                {"status": "estimated", "replicaeffective_samples": effective}
            )
    blocking = []
    for count in block_counts:
        block_means = x.reshape(m, count, n // count, p).mean(axis=2)
        blocking.append(
            {
                "block_count_per_replica": count,
                "frames_per_block": n // count,
                "replica_block_mean_standard_error": (
                    block_means.std(axis=1, ddof=1) / np.sqrt(count)
                ).tolist(),
            }
        )
    cumulative = [
        {
            "frames_per_replica": end,
            "mean": x[:, :end].mean(axis=(0, 1)).tolist(),
            "replica_means": x[:, :end].mean(axis=1).tolist(),
        }
        for end in range(n // 10, n + 1, n // 10)
    ]
    return {
        "replica_count": m,
        "frames_per_replica": n,
        "mean": means.mean(axis=0).tolist(),
        "replica_means": means.tolist(),
        "replica_mean_standard_error": (
            means.std(axis=0, ddof=1) / np.sqrt(m)
        ).tolist(),
        "split_rhat": rhat,
        "effective_samples": ess,
        "diagnostics": diagnostics,
        "blocking": blocking,
        "cumulative": cumulative,
        "second_minus_first_half": (
            x[:, n // 2 :].mean(axis=1) - x[:, : n // 2].mean(axis=1)
        ).tolist(),
    }


def check_sampling(
    statistics: dict[str, Any],
    names: list[str],
    *,
    split_rhat_max: float,
    combined_ess_min: float,
    required: int,
) -> dict[str, Any]:
    """Judge each quantity's split R-hat and effective samples against limits.

    A quantity is `pass` when both limits hold, `fail` when either is missed or
    the series is constant (nothing was sampled), and `unassessed` when the
    diagnostics could not be estimated. The whole is `pass` when at least
    `required` quantities pass, `unassessed` when unassessed ones could still
    bring it there, and `fail` otherwise.
    """
    rows = []
    for name, rhat, ess, diagnostic in zip(
        names,
        statistics["split_rhat"],
        statistics["effective_samples"],
        statistics["diagnostics"],
        strict=True,
    ):
        state = diagnostic["status"]
        if state in {"constant", "zero-within-chain-variance"}:
            verdict = "fail"
        elif state != "estimated":
            verdict = "unassessed"
        else:
            verdict = (
                "pass" if rhat <= split_rhat_max and ess >= combined_ess_min else "fail"
            )
        rows.append(
            {
                "id": name,
                "diagnostic": state,
                "split_rhat": rhat,
                "effective_samples": ess,
                "status": verdict,
            }
        )
    passed = sum(r["status"] == "pass" for r in rows)
    missing = sum(r["status"] == "unassessed" for r in rows)
    status = (
        "pass"
        if passed >= required
        else "unassessed"
        if passed + missing >= required
        else "fail"
    )
    return {
        "split_rhat_max": split_rhat_max,
        "combined_ess_min": combined_ess_min,
        "required_jointly_passing": required,
        "jointly_passing": passed,
        "unassessed": missing,
        "status": status,
        "quantities": rows,
    }
