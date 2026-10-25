"""Replicated one-dimensional umbrella densities, reconstructed with MBAR.

Runs inside the analysis environment (PyMBAR, NumPy). Windows share one Hamiltonian and
one temperature and differ by a harmonic bias K (q - center)^2 on a reaction coordinate.
Each window's coordinate series is decorrelated with a conservative stride, the windows
are combined with MBAR against the unbiased state, and the density of the coordinate in
the bins you set gives a potential of mean force with its asymptotic covariance. Basin
free-energy differences, the peak over a transition region and, if asked for, a classical
one-way flux through a dividing surface follow, for each independent replica and for
chronological blocks of each. A second profile reweights the same samples to a target
Hamiltonian from paired potential energies.
"""

import math
import re
from importlib.metadata import version

PROFILE = "pymbar.umbrella-distance-difference.v1"
TARGET_PROFILE = "pymbar.umbrella-target-reweighting.v1"
VERSION = "4.2.0"
GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError(f"{name} requires exactly these fields: {fields}")


def _number(value, name, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{name} requires a finite {'positive ' if positive else ''}number"
        )


def _id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty string")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Source identities require lowercase SHA-256 digests")


def validate_parameters(p):
    _fields(
        p,
        "bin_edges_angstrom reference_bin reactant_bins product_bins transition_bins block_counts"
        + (
            " transition_state_flux"
            if isinstance(p, dict) and "transition_state_flux" in p
            else ""
        ),
        "Umbrella analysis",
    )
    edges = p["bin_edges_angstrom"]
    if not isinstance(edges, list) or not 4 <= len(edges) <= 201:
        raise ValueError("Supply 3–200 histogram bins")
    for edge in edges:
        _number(edge, "Bin edge")
    if any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError("Bin edges require strictly increasing coordinates")
    seen = set()
    for key in ("reactant_bins", "product_bins", "transition_bins"):
        bins = p[key]
        if (
            not isinstance(bins, list)
            or not bins
            or any(type(b) is not int or not 0 <= b < len(edges) - 1 for b in bins)
        ):
            raise ValueError(f"{key} requires explicit zero-based histogram bins")
        if len(set(bins)) != len(bins) or seen.intersection(bins):
            raise ValueError("Reactant, product and transition bins must be distinct")
        seen.update(bins)
    if (
        type(p["reference_bin"]) is not int
        or p["reference_bin"] not in p["reactant_bins"]
    ):
        raise ValueError("The explicit reference bin must belong to the reactant basin")
    blocks = p["block_counts"]
    if (
        not isinstance(blocks, list)
        or not 1 <= len(blocks) <= 4
        or any(type(b) is not int or not 2 <= b <= 10 for b in blocks)
        or len(set(blocks)) != len(blocks)
    ):
        raise ValueError("Supply 1–4 distinct time-block counts between 2 and 10")
    if "transition_state_flux" in p:
        flux = p["transition_state_flux"]
        _fields(
            flux,
            "transition_bin inverse_mass_dalton",
            "Classical transition-state flux",
        )
        if (
            type(flux["transition_bin"]) is not int
            or flux["transition_bin"] not in p["transition_bins"]
        ):
            raise ValueError(
                "The explicit dividing-surface bin must belong to the transition region"
            )
        _number(
            flux["inverse_mass_dalton"],
            "Constant inverse coordinate mass in dalton^-1",
            positive=True,
        )


def validate_umbrella(data, p, profile=PROFILE):
    validate_parameters(p)
    if profile not in (PROFILE, TARGET_PROFILE):
        raise ValueError("Select an implemented umbrella analysis profile")
    paired = profile == TARGET_PROFILE
    schema = "0.2" if paired else "0.1"
    if not isinstance(data, dict) or data.get("schema_version") != schema:
        raise ValueError(f"{profile} requires umbrella schema {schema}")
    _fields(
        data,
        "schema_version title system_sha256 hamiltonian_sha256 temperature_kelvin coordinate bias_convention windows replicas"
        + (" target" if paired else ""),
        "Umbrella samples",
    )
    if data["bias_convention"] != "K*(q-center)^2":
        raise ValueError("Use the explicit K*(q-center)^2 bias convention")
    if paired:
        target = data["target"]
        _fields(
            target,
            "hamiltonian_sha256 energy_unit energy_convention configuration_measure",
            "Target Hamiltonian",
        )
        _digest(target["hamiltonian_sha256"])
        if (
            target["energy_unit"] != "kJ/mol"
            or target["energy_convention"] != "unbiased-potential-energy"
            or target["configuration_measure"]
            != "same-positions-cell-composition-and-constraints"
        ):
            raise ValueError(
                "Paired unbiased potential energies use kJ/mol and identical positions, cell, composition and constraints"
            )
    _id(data["title"], "Title")
    for key in ("system_sha256", "hamiltonian_sha256"):
        _digest(data[key])
    _number(data["temperature_kelvin"], "Temperature", positive=True)
    coordinate = data["coordinate"]
    _fields(
        coordinate,
        "unit definition positive_distance_atoms negative_distance_atoms",
        "Reaction coordinate",
    )
    if coordinate["unit"] != "angstrom":
        raise ValueError("Reaction-coordinate distances use angstrom")
    pairs = []
    for key in ("positive_distance_atoms", "negative_distance_atoms"):
        pair = coordinate[key]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(i) is not int or i < 1 for i in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError(
                "Distance endpoints require two distinct one-based atom indices"
            )
        pairs.append(pair)
    if set(pairs[0]) == set(pairs[1]):
        raise ValueError("The two distances must have distinct endpoint pairs")
    expected = (
        f"distance({pairs[0][0]},{pairs[0][1]})-distance({pairs[1][0]},{pairs[1][1]})"
    )
    if coordinate["definition"] != expected:
        raise ValueError(
            "Coordinate definition must exactly match its ordered atom pairs"
        )
    if "transition_state_flux" in p and len(set(pairs[0] + pairs[1])) != 4:
        raise ValueError(
            "Constant distance-difference mass metric requires four distinct flexible atoms"
        )
    windows = data["windows"]
    if not isinstance(windows, list) or not 2 <= len(windows) <= 100:
        raise ValueError("Supply 2–100 harmonic windows sharing the stated Hamiltonian")
    identities, centers = [], []
    for window in windows:
        _fields(window, "id center_angstrom k_kj_mol_angstrom2", "Umbrella window")
        _id(window["id"], "Window ID")
        _number(window["center_angstrom"], "Window center")
        _number(window["k_kj_mol_angstrom2"], "Restraint coefficient", positive=True)
        identities.append(window["id"])
        centers.append(window["center_angstrom"])
    if len(set(identities)) != len(identities) or any(
        b <= a for a, b in zip(centers[:-1], centers[1:], strict=True)
    ):
        raise ValueError("Window IDs are unique and centers strictly increasing")
    replicas = data["replicas"]
    if not isinstance(replicas, list) or not 2 <= len(replicas) <= 20:
        raise ValueError("Supply 2–20 independently sampled complete umbrella replicas")
    seen, digests, total = set(), set(), 0
    for replica in replicas:
        _fields(replica, "id windows", "Replica")
        _id(replica["id"], "Replica ID")
        if replica["id"] in seen:
            raise ValueError("Replica identities must be distinct")
        seen.add(replica["id"])
        traces = replica["windows"]
        if not isinstance(traces, list) or len(traces) != len(windows):
            raise ValueError("Every replica must contain every window in source order")
        replica_count = 0
        for expected_id, trace in zip(identities, traces, strict=True):
            _fields(
                trace,
                "window trajectory_sha256 discard_frames frame_interval_fs coordinate_angstrom"
                + (" energy_pairs" if paired else ""),
                "Window trajectory",
            )
            _digest(trace["trajectory_sha256"])
            if trace["window"] != expected_id or trace["trajectory_sha256"] in digests:
                raise ValueError(
                    "Window order must match and trajectory identities must be distinct"
                )
            digests.add(trace["trajectory_sha256"])
            _number(trace["frame_interval_fs"], "Frame interval", positive=True)
            q = trace["coordinate_angstrom"]
            if not isinstance(q, list) or not q:
                raise ValueError("Coordinate trace requires a nonempty numeric list")
            for value in q:
                _number(value, "Sampled coordinate")
            if paired:
                energy = trace["energy_pairs"]
                _fields(
                    energy,
                    "reference_source_sha256 target_source_sha256 reference_frame_sha256 target_frame_sha256 reference_kj_mol target_kj_mol",
                    "Paired energy records",
                )
                for side in ("reference", "target"):
                    _digest(energy[f"{side}_source_sha256"])
                    for suffix, check in (
                        ("frame_sha256", _digest),
                        ("kj_mol", lambda v: _number(v, "Paired energy")),
                    ):
                        values = energy[f"{side}_{suffix}"]
                        if not isinstance(values, list) or len(values) != len(q):
                            raise ValueError(
                                "Each coordinate requires exactly one matching frame identity and energy pair"
                            )
                        for value in values:
                            check(value)
                if energy["reference_frame_sha256"] != energy["target_frame_sha256"]:
                    raise ValueError(
                        "Reference and target frame identities must match in source coordinate order"
                    )
            discard = trace["discard_frames"]
            if (
                type(discard) is not int
                or discard < 0
                or len(q) - discard < 20 * max(p["block_counts"])
            ):
                raise ValueError(
                    "Explicit discarded prefix must leave at least 20 frames in every time block"
                )
            replica_count += len(q)
        if replica_count > 100_000:
            raise ValueError(
                "Each analysis replica accepts at most 100,000 supplied frames"
            )
        total += replica_count
    if total > 1_000_000:
        raise ValueError("One analysis accepts at most 1,000,000 supplied frames")


def classical_flux(
    probability,
    covariance,
    widths_angstrom,
    reactant_bins,
    transition_bin,
    temperature,
    inverse_mass_dalton,
):
    """Evaluate an explicit finite-width surface density and integrated reactant mass.

    Nuclear momenta follow the classical canonical distribution. The positive
    velocity factor in A/ps is 10 sqrt(R T Z/(2 pi)), where Z is the inverse
    effective mass in dalton^-1. The rate is that factor times rho(surface)/P(A).
    The reported activation free energy uses the k_B T/h reference frequency.
    """
    import numpy as np

    p = np.asarray(probability, float)
    c = np.asarray(covariance, float)
    widths = np.asarray(widths_angstrom, float)
    if (
        p.ndim != 1
        or len(p) < 2
        or c.shape != (len(p), len(p))
        or widths.shape != p.shape
        or not all(np.isfinite(a).all() for a in [p, c, widths])
        or (p < 0).any()
        or p.sum() > 1 + 1e-8
        or (widths <= 0).any()
        or not np.isfinite([temperature, inverse_mass_dalton]).all()
        or min(temperature, inverse_mass_dalton) <= 0
        or not reactant_bins
        or len(set(reactant_bins)) != len(reactant_bins)
        or any(type(i) is not int or not 0 <= i < len(p) for i in reactant_bins)
        or type(transition_bin) is not int
        or not 0 <= transition_bin < len(p)
        or transition_bin in reactant_bins
    ):
        raise ValueError(
            "Finite probabilities/covariance, positive bin widths, disjoint explicit regions and positive thermal/mass inputs required"
        )
    scale = max(float(np.linalg.norm(c, ord=2)), np.finfo(float).tiny)
    if (
        not np.allclose(c, c.T, rtol=0, atol=1e-10 * scale)
        or np.linalg.eigvalsh(c).min() < -1e-10 * scale
    ):
        raise ValueError(
            "A symmetric positive-semidefinite bin-probability covariance is required"
        )
    pa = float(p[reactant_bins].sum())
    pt = float(p[transition_bin])
    width = float(widths[transition_bin])
    if min(pa, pt) <= 0:
        raise ValueError(
            "Positive sampled reactant and transition probabilities required"
        )
    rt = GAS_CONSTANT_KJ_MOL_K * temperature
    velocity = 10 * math.sqrt(rt * inverse_mass_dalton / (2 * math.pi))
    log_rate = (
        math.log(pt)
        - math.log(width)
        - math.log(pa)
        + math.log(velocity)
        + math.log(1e12)
    )
    try:
        rate = math.exp(log_rate)
    except OverflowError as error:
        raise ValueError("A finite positive rate representation is required") from error
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("A finite positive rate representation is required")
    gradient = np.zeros_like(p)
    gradient[reactant_bins] = -1 / pa
    gradient[transition_bin] = 1 / pt
    variance = float(gradient @ c @ gradient)
    if not math.isfinite(variance) or variance < -1e-10 * max(
        scale * float(gradient @ gradient), np.finfo(float).tiny
    ):
        raise ValueError("Resolved finite propagated log-rate variance required")
    error = math.sqrt(max(0, variance))
    frequency = 1.380649e-23 * temperature / 6.62607015e-34
    return {
        "reactant_probability": pa,
        "transition_bin_probability": pt,
        "transition_density_per_angstrom": pt / width,
        "transition_bin_width_angstrom": width,
        "positive_velocity_angstrom_per_ps": velocity,
        "inverse_mass_dalton": inverse_mass_dalton,
        "temperature_kelvin": temperature,
        "rate_per_second": rate,
        "log_rate_per_second": log_rate,
        "log_rate_asymptotic_error": error,
        "eyring_reference_frequency_per_second": frequency,
        "activation_free_energy_kj_mol": -rt * (log_rate - math.log(frequency)),
        "activation_free_energy_asymptotic_error_kj_mol": rt * error,
        "convention": "Classical canonical transition-state one-way flux; finite-width surface-density estimate; transmission factor 1; probability covariance propagated for the explicitly selected surface and reactant basin.",
    }


def _sample(traces, windows, fraction):
    import numpy as np
    from pymbar import timeseries

    selected, diagnostics = [], []
    for trace, window in zip(traces, windows, strict=True):
        start = trace["discard_frames"]
        length = len(trace["coordinate_angstrom"]) - start
        lo, hi = start + int(length * fraction[0]), start + int(length * fraction[1])
        q = np.asarray(trace["coordinate_angstrom"][lo:hi], dtype=float)
        if len(q) < 20 or np.ptp(q) == 0:
            raise ValueError(
                f"{trace['window']}: sampling requires >=20 frames and a coordinate that varies"
            )
        bias = window["k_kj_mol_angstrom2"] * (q - window["center_angstrom"]) ** 2
        gq = float(timeseries.statistical_inefficiency(q))
        gbias = (
            float(timeseries.statistical_inefficiency(bias)) if np.ptp(bias) else 1.0
        )
        g = max(gq, gbias)
        energy_diagnostics = {}
        if "energy_pairs" in trace:
            energy = trace["energy_pairs"]
            reference = np.asarray(energy["reference_kj_mol"][lo:hi], dtype=float)
            delta = np.asarray(energy["target_kj_mol"][lo:hi], dtype=float) - reference
            if not np.isfinite(delta).all():
                raise ValueError("Paired energy differences require finite values")
            for name, values in (
                ("reference_energy", reference),
                ("energy_difference", delta),
            ):
                inefficiency = (
                    float(timeseries.statistical_inefficiency(values))
                    if np.ptp(values)
                    else 1.0
                )
                energy_diagnostics[f"statistical_inefficiency_{name}"] = inefficiency
                if not math.isfinite(inefficiency) or inefficiency < 1:
                    raise ValueError(
                        "Native energy decorrelation returned an invalid statistical inefficiency"
                    )
                g = max(g, inefficiency)
        if not math.isfinite(g) or g < 1:
            raise ValueError(
                "Native decorrelation returned an invalid statistical inefficiency"
            )
        indices = timeseries.subsample_correlated_data(q, g=g, conservative=True)
        if len(indices) < 5:
            raise ValueError(
                f"{trace['window']}: fewer than five decorrelated samples remain"
            )
        selected.append(q[indices])
        diagnostics.append(
            {
                "window": window["id"],
                "first_source_frame": lo,
                "end_source_frame_exclusive": hi,
                "statistical_inefficiency_coordinate": gq,
                "statistical_inefficiency_bias": gbias,
                "stride_frames": math.ceil(g),
                "selected_samples": len(indices),
                "selected_source_frame_indices": [lo + int(i) for i in indices],
                **energy_diagnostics,
            }
        )
    return selected, diagnostics


def _paired_differences(traces, diagnostics):
    if "energy_pairs" not in traces[0]:
        return None
    return [
        trace["energy_pairs"]["target_kj_mol"][i]
        - trace["energy_pairs"]["reference_kj_mol"][i]
        for trace, diagnostic in zip(traces, diagnostics, strict=True)
        for i in diagnostic["selected_source_frame_indices"]
    ]


def _weight_diagnostics(weights):
    import numpy as np

    total = float(weights.sum())
    if total <= 0:
        return {
            "samples": len(weights),
            "effective_samples": 0.0,
            "maximum_weight": None,
            "normalized_entropy": None,
        }
    normalized = weights / total
    positive = normalized[normalized > 0]
    return {
        "samples": len(weights),
        "effective_samples": float(1 / np.sum(normalized**2)),
        "maximum_weight": float(normalized.max()),
        "normalized_entropy": float(
            -np.sum(positive * np.log(positive)) / math.log(len(weights))
        )
        if len(weights) > 1
        else 0.0,
    }


def _estimate(samples, windows, temperature, p, target_differences=None):
    import numpy as np
    from pymbar import MBAR

    rt = GAS_CONSTANT_KJ_MOL_K * temperature
    q = np.concatenate(samples)
    centers = np.array([w["center_angstrom"] for w in windows])
    coefficients = np.array([w["k_kj_mol_angstrom2"] for w in windows])
    # The sampling Hamiltonian cancels column-wise. The zero row is the
    # unsampled, umbrella-free reference; a paired target adds its energy change.
    u_kn = np.vstack(
        [coefficients[:, None] * (q - centers[:, None]) ** 2 / rt, np.zeros(len(q))]
    )
    counts = np.array([len(s) for s in samples] + [0])
    target_u = np.zeros(len(q))
    paired = target_differences is not None
    if paired:
        delta = np.asarray(target_differences, dtype=float)
        if delta.shape != q.shape or not np.isfinite(delta).all():
            raise ValueError(
                "Every retained sample requires a finite paired energy difference"
            )
        offset = float(np.min(delta))
        target_u = (delta - offset) / rt
        if not np.isfinite(target_u).all():
            raise ValueError("Reduced paired energy differences require finite values")
        u_kn = np.vstack([u_kn, target_u])
        counts = np.append(counts, 0)
    mbar = MBAR(u_kn, counts, relative_tolerance=1e-10, solver_protocol="robust")
    overlap = mbar.compute_overlap()["matrix"][: len(windows), : len(windows)]
    reached = {0}
    while True:
        connected = {
            j
            for i in reached
            for j in range(len(windows))
            if max(overlap[i, j], overlap[j, i]) > 1e-12
        }
        if connected <= reached:
            break
        reached.update(connected)
    if len(reached) != len(windows):
        raise ValueError(
            "Umbrella overlap is numerically disconnected; relative normalization requires connected sampling"
        )
    edges = np.array(p["bin_edges_angstrom"])
    n_bins = len(edges) - 1
    bins = np.searchsorted(edges, q, side="right") - 1
    bins[q == edges[-1]] = n_bins - 1
    occupied = [i for i in range(n_bins) if np.any(bins == i)]
    if p["reference_bin"] not in occupied:
        raise ValueError("The specified reference bin has zero sampled occupancy")
    indicators = np.array([bins == i for i in occupied], dtype=float)
    # Zero indicator values give legitimate -inf log weights in native MBAR.
    with np.errstate(divide="ignore"):
        native = mbar.compute_multiple_expectations(
            indicators, target_u, return_theta=True
        )
    mass = np.asarray(native["mu"])
    # PyMBAR 4.2.0's `covariances` field returns its internal log-weight block.
    # `Theta` is the native observable-scaled matrix; use the same four-block
    # contraction the upstream routine uses to produce its validated `sigma`.
    theta, n = np.asarray(native["Theta"]), len(occupied)
    covariance = theta[:n, :n] + theta[n:, n:] - theta[:n, n:] - theta[n:, :n]
    if not np.allclose(
        np.diag(covariance), native["sigma"] ** 2, rtol=1e-8, atol=1e-12
    ):
        raise ValueError(
            "Observable covariance must reproduce native expectation uncertainties"
        )
    if (
        not np.isfinite(mass).all()
        or np.any(mass <= 0)
        or not np.isfinite(covariance).all()
    ):
        raise ValueError(
            "Native bin probabilities and covariance require finite positive sampled mass"
        )
    probability_products = np.outer(mass, mass)
    if np.any(probability_products < np.finfo(float).tiny):
        raise ValueError(
            "Bin probabilities require normal-range covariance products; inspect energy differences and target-weight concentration"
        )
    density = mass / np.diff(edges)[occupied]
    ref = occupied.index(p["reference_bin"])
    f = -rt * np.log(density / density[ref])
    log_covariance = covariance / probability_products
    variance = rt**2 * (
        np.diag(log_covariance) + log_covariance[ref, ref] - 2 * log_covariance[:, ref]
    )
    if np.min(variance) < -1e-8:
        raise ValueError("Native propagated PMF variance is numerically negative")
    errors = np.sqrt(np.maximum(variance, 0))
    f[ref], errors[ref] = 0.0, 0.0
    weights = mbar.W_nk[:, -1]
    ess = [
        float(weights[bins == i].sum() ** 2 / np.sum(weights[bins == i] ** 2))
        if i in occupied
        else 0.0
        for i in range(n_bins)
    ]

    def expand(values):
        mapped = dict(zip(occupied, values, strict=True))
        return [float(mapped[i]) if i in mapped else None for i in range(n_bins)]

    basin_masks = [
        np.array([i in p[key] for i in occupied], dtype=float)
        for key in ("reactant_bins", "product_bins")
    ]
    basin_mass = np.array([mask @ mass for mask in basin_masks])
    basin_delta, basin_error = None, None
    if np.all(basin_mass > 0):
        basin_delta = float(-rt * np.log(basin_mass[1] / basin_mass[0]))
        gradient = rt * (
            basin_masks[0] / basin_mass[0] - basin_masks[1] / basin_mass[1]
        )
        basin_variance = float(gradient @ covariance @ gradient)
        if basin_variance < -1e-8:
            raise ValueError("Native propagated basin variance is numerically negative")
        basin_error = math.sqrt(max(0, basin_variance))
    pmf = expand(f)
    transition = p["transition_bins"]
    peak_bin = (
        max(transition, key=lambda i: pmf[i])
        if all(pmf[i] is not None for i in transition)
        else None
    )
    result = {
        "pmf_kj_mol": pmf,
        "pmf_asymptotic_error_kj_mol": expand(errors),
        "bin_probability": [v if v is not None else 0.0 for v in expand(mass)],
        "bin_probability_covariance": [
            [
                float(covariance[occupied.index(i), occupied.index(j)])
                if i in occupied and j in occupied
                else 0.0
                for j in range(n_bins)
            ]
            for i in range(n_bins)
        ],
        "bin_effective_samples": ess,
        "outside_probability": float(weights[(bins < 0) | (bins >= n_bins)].sum()),
        "basin_delta_g_kj_mol": basin_delta,
        "basin_asymptotic_error_kj_mol": basin_error,
        "reactant_probability": float(basin_mass[0]),
        "product_probability": float(basin_mass[1]),
        "transition_peak_bin": peak_bin,
        "peak_minus_reference_kj_mol": pmf[peak_bin] if peak_bin is not None else None,
        "overlap_matrix": overlap.tolist(),
        "state_effective_samples": mbar.compute_effective_sample_number()[
            : len(windows)
        ].tolist(),
        "target_effective_samples": float(mbar.compute_effective_sample_number()[-1]),
    }
    if "transition_state_flux" in p:
        specification = p["transition_state_flux"]
        if (
            result["reactant_probability"] <= 0
            or result["bin_probability"][specification["transition_bin"]] <= 0
        ):
            result["transition_state_flux"] = {
                "estimate": None,
                "error": "Positive sampled probability in the explicit reactant basin and dividing-surface bin is required",
            }
        else:
            result["transition_state_flux"] = {
                "estimate": classical_flux(
                    result["bin_probability"],
                    result["bin_probability_covariance"],
                    np.diff(edges),
                    p["reactant_bins"],
                    specification["transition_bin"],
                    temperature,
                    specification["inverse_mass_dalton"],
                ),
                "error": None,
            }
    if paired:
        reference_ess = float(mbar.compute_effective_sample_number()[-2])
        result["reweighting"] = {
            **_weight_diagnostics(weights),
            "reference_effective_samples": reference_ess,
            "target_reference_ess_ratio": result["target_effective_samples"]
            / reference_ess,
            "subtracted_delta_energy_kj_mol": offset,
            "bins": [_weight_diagnostics(weights[bins == i]) for i in range(n_bins)],
        }
    return result


def analyze_umbrella(data, parameters, profile=PROFILE):
    """Reconstruct a density PMF, optionally at a paired target Hamiltonian."""
    import numpy as np
    from pymbar.utils import BoundsError, ConvergenceError, DataError, ParameterError

    validate_umbrella(data, parameters, profile)
    if version("pymbar") != VERSION:
        raise ValueError(f"{profile} requires PyMBAR {VERSION}")
    replicas = []
    for replica in data["replicas"]:
        samples, diagnostic = _sample(replica["windows"], data["windows"], (0, 1))
        result = _estimate(
            samples,
            data["windows"],
            data["temperature_kelvin"],
            parameters,
            _paired_differences(replica["windows"], diagnostic),
        )
        result.update(id=replica["id"], sampling=diagnostic, time_blocks=[])
        for count in parameters["block_counts"]:
            blocks = []
            for i in range(count):
                try:
                    subset, retained = _sample(
                        replica["windows"],
                        data["windows"],
                        (i / count, (i + 1) / count),
                    )
                    estimate = _estimate(
                        subset,
                        data["windows"],
                        data["temperature_kelvin"],
                        parameters,
                        _paired_differences(replica["windows"], retained),
                    )
                    blocks.append(
                        {
                            "index": i,
                            "sampling": retained,
                            "estimate": estimate,
                            "error": None,
                        }
                    )
                except (
                    ValueError,
                    BoundsError,
                    ConvergenceError,
                    DataError,
                    ParameterError,
                    np.linalg.LinAlgError,
                ) as exc:
                    blocks.append({"index": i, "estimate": None, "error": str(exc)})
            result["time_blocks"].append({"count": count, "blocks": blocks})
        replicas.append(result)

    def summary(values):
        if any(v is None for v in values):
            return None, None, None
        sd = float(np.std(values, ddof=1))
        return float(np.mean(values)), sd, sd / math.sqrt(len(values))

    means, sds, sems = zip(
        *(
            summary([r["pmf_kj_mol"][i] for r in replicas])
            for i in range(len(parameters["bin_edges_angstrom"]) - 1)
        ),
        strict=True,
    )
    basin = summary([r["basin_delta_g_kj_mol"] for r in replicas])
    result = {
        "schema_version": data["schema_version"],
        "title": data["title"],
        "method_profile": profile,
        "system_sha256": data["system_sha256"],
        "hamiltonian_sha256": data["hamiltonian_sha256"],
        "temperature_kelvin": data["temperature_kelvin"],
        "coordinate": data["coordinate"],
        "bias_convention": data["bias_convention"],
        "windows": data["windows"],
        "parameters": parameters,
        "replicas": replicas,
        "method": {
            "pymbar_version": VERSION,
            "gas_constant_kj_mol_k": GAS_CONSTANT_KJ_MOL_K,
            "unbiasing": "Common-Hamiltonian, common-temperature equilibrium windows; remove K*(q-center)^2 only",
            "density": "Probability per angstrom; each bin probability divided by its own width",
            "reference": "Explicit reactant reference bin; identical gauge across replicas and time blocks",
            "uncertainty": "Native asymptotic covariance after coordinate/bias decorrelation; independent-replica SD and SEM reported separately",
            "decorrelation": "Stride set by the larger native statistical inefficiency of the coordinate and the harmonic-bias energy",
            "time_blocks": "Chronological post-discard segments, independently decorrelated and reconstructed; retained segment failures",
            "basins": "Negative RT log of product-to-reactant integrated probability, including bin covariance",
            "peak": "Maximum populated transition-bin density PMF relative to the stated reference bin",
            "interpretation": "Coordinate-density free energy under the stated common Hamiltonian, including its other restraints",
        },
        "summary": {
            "replicas": len(replicas),
            "pmf_mean_kj_mol": list(means),
            "pmf_replica_sd_kj_mol": list(sds),
            "pmf_replica_sem_kj_mol": list(sems),
            "basin_delta_g_mean_kj_mol": basin[0],
            "basin_delta_g_replica_sd_kj_mol": basin[1],
            "basin_delta_g_replica_sem_kj_mol": basin[2],
        },
    }
    if "transition_state_flux" in parameters:
        result["method"]["transition_state_flux"] = (
            "Explicit finite-width dividing surface and integrated reactant basin; classical canonical momenta, "
            "constant supplied inverse coordinate mass, transmission factor 1. The Eyring reference is k_B*T/h; "
            "probability covariance propagates to log rate and activation free energy. Shared molecular restraints remain included."
        )
        flux_summary = {}
        for key in ("log_rate_per_second", "activation_free_energy_kj_mol"):
            estimates = [r["transition_state_flux"]["estimate"] for r in replicas]
            mean, sd, sem = summary(
                [e[key] if e is not None else None for e in estimates]
            )
            flux_summary[key] = {"mean": mean, "replica_sd": sd, "replica_sem": sem}
        result["summary"]["transition_state_flux"] = flux_summary
    if profile == TARGET_PROFILE:
        result["sampling_hamiltonian_sha256"] = data["hamiltonian_sha256"]
        result["hamiltonian_sha256"] = data["target"]["hamiltonian_sha256"]
        result["target"] = data["target"]
        result["method"].update(
            unbiasing="Common-temperature reference umbrella windows; target reduced potential is (U_target-U_reference)/RT after removing K*(q-center)^2",
            uncertainty="Native asymptotic covariance after coordinate/bias/reference-energy/energy-difference decorrelation; independent-replica SD and SEM reported separately",
            decorrelation="Stride set by the largest native statistical inefficiency of the coordinate, harmonic bias, reference energy and paired energy difference",
            interpretation="Coordinate-density free energy under the stated target Hamiltonian on the same configuration measure, including its other restraints",
            reweighting="Normalized target-frame weights; Kish effective samples, maximum weight and Shannon entropy divided by log(sample count), globally and within each bin",
            energy_offset="One constant energy-difference offset per estimate is subtracted for numerical conditioning; normalized probabilities are invariant",
        )
    return result


def run(cfg):
    """Engine entry: analyse the supplied umbrella samples."""
    return analyze_umbrella(cfg["samples"], cfg["parameters"], cfg["method_profile"])
