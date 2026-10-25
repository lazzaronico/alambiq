"""Molecular PBE/6-31G** energies and full Gaussian-embedding derivatives.

The commissioned closed-shell path uses PySCF 2.14.0 on the CPU. Coordinates and Gaussian radii use
angstrom; energies and forces retain native hartree and hartree/bohr units.
"""

import json
import math
import time
from pathlib import Path


def validate_inputs(
    elements,
    quantum_xyz,
    embedding_xyz,
    embedding_charges,
    *,
    charge,
    grid,
    gaussian_radius_angstrom,
    backend,
    threads,
    memory_mb,
):
    import numpy as np

    q, m, c = (
        np.asarray(v, float) for v in (quantum_xyz, embedding_xyz, embedding_charges)
    )
    if (
        not isinstance(elements, (list, tuple))
        or not elements
        or any(type(e) is not int or not 1 <= e <= 118 for e in elements)
        or q.shape != (len(elements), 3)
        or m.ndim != 2
        or m.shape[1] != 3
        or not len(m)
        or c.shape != (len(m),)
        or not all(np.isfinite(a).all() for a in (q, m, c))
    ):
        raise ValueError(
            "Resolved atomic numbers, finite quantum/embedding coordinates and matching charges are required"
        )
    if (
        type(charge) is not int
        or sum(elements) - charge <= 0
        or (sum(elements) - charge) % 2
        or type(grid) is not int
        or not 0 <= grid <= 9
        or type(gaussian_radius_angstrom) not in (int, float)
        or not math.isfinite(gaussian_radius_angstrom)
        or gaussian_radius_angstrom <= 0
        or backend != "cpu"
        or type(threads) is not int
        or threads <= 0
        or type(memory_mb) is not int
        or memory_mb <= 0
    ):
        raise ValueError(
            "A positive closed-shell population, explicit quadrature/backend and positive resource settings are required"
        )
    return {
        "electrons": sum(elements) - charge,
        "quantum": q.copy(),
        "embedding": m.copy(),
        "charges": c.copy(),
    }


def stationary(converged, energy, population, electrons, gradient):
    return bool(
        converged
        and all(math.isfinite(x) for x in (energy, population, gradient))
        and abs(population - electrons) <= 1e-7
        and 0 <= gradient <= 1e-7
    )


def condition_diis(method):
    """Normalize the residual Gram block while retaining native DIIS solving."""
    import numpy as np

    original = method.DIIS
    scales = []

    class RelativeErrorDIIS(original):
        def extrapolate(self, nd=None):
            count = self.get_num_vec() if nd is None else nd
            if count == 0:
                return super().extrapolate(count)
            block = self._H[1 : count + 1, 1 : count + 1]
            saved = block.copy()
            scale = float(np.max(np.abs(saved)))
            if not np.isfinite(scale):
                raise ValueError("A finite DIIS residual metric is required")
            scales.append(scale)
            try:
                if scale > 0:
                    block[:] = saved / scale
                return super().extrapolate(count)
            finally:
                block[:] = saved

    method.DIIS = RelativeErrorDIIS
    return scales


def full_density_updates(method):
    """Rebuild the native Coulomb potential from the complete current density."""
    native = method.get_veff

    def evaluate(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, **kwargs):
        return native(mol=mol, dm=dm, dm_last=0, vhf_last=0, hermi=hermi, **kwargs)

    method.get_veff = evaluate
    return method


def _host(value):
    import numpy as np

    return np.asarray(value.get() if hasattr(value, "get") else value)


def _record(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def gradient_components(gradient, method, backend, directory):
    """Observe one native gradient call and retain comparable atomic contributions."""
    import numpy as np

    arrays, saved, counts = {}, {}, {}
    shape = (method.mol.natm, 3)

    def retain(name, value):
        value = np.asarray(_host(value), dtype=float).copy()
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Finite atom-by-component native gradients required")
        if name in arrays:
            raise ValueError("Record each complete native gradient contribution once")
        arrays[name] = value

    def contract(matrix):
        matrix = _host(matrix)
        density = _host(method.make_rdm1())
        return np.asarray(
            [
                2 * np.einsum("xij,ij->x", matrix[:, p0:p1], density[p0:p1])
                for _, _, p0, p1 in method.mol.aoslice_by_atom()
            ]
        )

    def observe(name, capture):
        native = getattr(gradient, name)
        saved[name] = (name in gradient.__dict__, gradient.__dict__.get(name))

        def call(*args, **kwargs):
            value = native(*args, **kwargs)
            counts[name] = counts.get(name, 0) + 1
            capture(value, args, kwargs)
            return value

        setattr(gradient, name, call)

    def whole(label):
        return lambda value, args, kwargs: retain(label, value)

    try:
        observe("grad_elec", whole("electronic"))
        observe("grad_nuc", whole("nuclear"))
        if backend != "cpu":
            raise ValueError("The molecular gradient backend is the CPU")
        observe("get_j", lambda value, args, kwargs: retain("coulomb", contract(value)))
        observe(
            "get_veff",
            lambda value, args, kwargs: retain("pair_fixed_grid", contract(value)),
        )
        response, atoms = np.zeros(shape), set()

        def extra(value, args, kwargs):
            atom = args[0] if args else kwargs["atom_id"]
            if (
                type(atom) not in (int, np.int32, np.int64)
                or not 0 <= atom < shape[0]
                or atom in atoms
            ):
                raise ValueError("Record each native atomic grid response once")
            response[atom] = np.broadcast_to(_host(value), (3,))
            atoms.add(atom)

        observe("extra_force", extra)
        result = gradient.kernel()
        retain("total", result)
        if atoms != set(range(shape[0])):
            raise ValueError("Every quantum atom requires its native grid response")
        retain("grid_response", response)
        retain("pair_with_grid_response", arrays["pair_fixed_grid"] + response)
        arrays["exchange_correlation_with_grid_response"] = (
            arrays["pair_with_grid_response"] - arrays["coulomb"]
        )
        arrays["one_electron_overlap_and_embedding"] = (
            arrays["electronic"] - arrays["pair_with_grid_response"]
        )
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Complete finite native gradient contributions required")
        np.savez(Path(directory) / "gradient-components-hartree-bohr.npz", **arrays)
        _record(
            Path(directory) / "gradient-components.json",
            dict(
                schema_version="0.1",
                backend=backend,
                quantity="energy gradient",
                unit="hartree/bohr",
                atom_order="declared quantum atom order",
                model="closed-shell PBE without exact exchange or dispersion",
                arrays=sorted(arrays),
                native_calls=counts,
                native_gradient_kernel_calls=1,
                additional_electronic_or_gradient_evaluations=0,
                maximum_total_composition_residual=float(
                    np.max(
                        np.abs(
                            arrays["total"] - arrays["electronic"] - arrays["nuclear"]
                        )
                    )
                ),
                interpretation="Atomic contributions describe the native gradient. Forces have the opposite sign. The one-electron contribution includes overlap and quantum/embedding electronic derivatives. The exchange-correlation contribution includes the native moving-grid response.",
            ),
        )
        return result
    finally:
        for name, (owned, value) in saved.items():
            if owned:
                setattr(gradient, name, value)
            else:
                delattr(gradient, name)


def evaluate_molecular(
    elements,
    quantum_xyz,
    embedding_xyz,
    embedding_charges,
    *,
    charge,
    grid,
    gaussian_radius_angstrom,
    backend,
    threads,
    memory_mb,
    directory,
    initial_density=None,
):
    """Evaluate one explicit molecular geometry and retain its native electronic state.

    Every invocation owns a fresh native calculation. An optional saved density
    seeds SCF; final population and orbital stationarity are checked independently.
    Each caller supplies a new output directory for native logs and exact arrays.
    """
    import numpy as np

    values = validate_inputs(
        elements,
        quantum_xyz,
        embedding_xyz,
        embedding_charges,
        charge=charge,
        grid=grid,
        gaussian_radius_angstrom=gaussian_radius_angstrom,
        backend=backend,
        threads=threads,
        memory_mb=memory_mb,
    )
    import pyscf
    from pyscf import dft, gto, lib, qmmm

    if pyscf.__version__ != "2.14.0":
        raise ValueError("This molecular electronic profile requires PySCF 2.14.0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    lib.num_threads(threads)
    started = time.perf_counter()
    mol = None
    try:
        mol = gto.M(
            atom=list(zip(elements, values["quantum"], strict=True)),
            unit="Angstrom",
            basis="6-31g**",
            charge=charge,
            spin=0,
            max_memory=memory_mb,
            verbose=4,
            output=str(directory / "native.log"),
        )
        density = None
        if initial_density is not None:
            density = np.asarray(initial_density, float)
            if (
                density.shape != (mol.nao, mol.nao)
                or not np.isfinite(density).all()
                or not np.allclose(density, density.T, rtol=0, atol=1e-10)
            ):
                raise ValueError(
                    "A finite symmetric starting density with the exact basis dimension is required"
                )
            np.save(directory / "starting-density.npy", density)
        radii = np.full(len(values["charges"]), gaussian_radius_angstrom)
        method = qmmm.mm_charge(
            dft.RKS(mol),
            values["embedding"],
            values["charges"],
            radii=radii,
            unit="Angstrom",
        )
        full_density_updates(method)
        method.xc, method.grids.level = "PBE", grid
        method.conv_tol, method.conv_tol_grad, method.max_cycle = 1e-11, 1e-9, 200
        diis_scales = condition_diis(method)
        ready = time.perf_counter()
        energy = float(method.kernel(dm0=density))
        scf_done = time.perf_counter()
        density, overlap = _host(method.make_rdm1()), _host(method.get_ovlp())
        population = float(np.einsum("ij,ji", density, overlap))
        fock = method.get_fock(dm=method.make_rdm1())
        gradient_norm = float(
            np.linalg.norm(_host(method.get_grad(method.mo_coeff, method.mo_occ, fock)))
        )
        state = dict(
            density=density,
            overlap=overlap,
            mo_coeff=_host(method.mo_coeff),
            mo_occ=_host(method.mo_occ),
            mo_energy=_host(method.mo_energy),
        )
        np.savez(directory / "electronic-state.npz", **state)
        observation = dict(
            energy_hartree=energy,
            electrons=population,
            expected_electrons=values["electrons"],
            orbital_gradient_norm=gradient_norm,
            converged=bool(method.converged),
            stationarity_passed=stationary(
                method.converged, energy, population, values["electrons"], gradient_norm
            ),
            scf_cycles=int(method.cycles),
            diis_conditioning={
                "method": "unit-maximum-residual-gram-v1",
                "scales": diis_scales,
            },
            basis_functions=mol.nao,
            method="PBE",
            basis="6-31g**",
            grid=grid,
            charge=charge,
            spin=0,
            gaussian_radius_angstrom=gaussian_radius_angstrom,
            backend=backend,
            scf_energy_tolerance=1e-11,
            scf_gradient_tolerance=1e-9,
            final_gradient_limit=1e-7,
            pyscf_version=pyscf.__version__,
            threads=threads,
            memory_mb=memory_mb,
            setup_seconds=ready - started,
            scf_seconds=scf_done - ready,
        )
        if not all(np.isfinite(a).all() for a in state.values()):
            raise ValueError("A complete finite native electronic state is required")
        _record(directory / "electronic-check.json", observation)
        if not observation["stationarity_passed"]:
            raise ValueError(
                "Native convergence, exact population and directly observed orbital stationarity are required"
            )
        gradient_start = time.perf_counter()
        gradient = method.nuc_grad_method()
        gradient.grid_response = True
        fq = -_host(gradient_components(gradient, method, backend, directory))
        fm = -_host(gradient.grad_hcore_mm(method.make_rdm1()) + gradient.grad_nuc_mm())
        if (
            fq.shape != values["quantum"].shape
            or fm.shape != values["embedding"].shape
            or not np.isfinite(fq).all()
            or not np.isfinite(fm).all()
        ):
            raise ValueError(
                "Every finite native quantum and embedding force is required"
            )
        completed = time.perf_counter()
        observation.update(
            gradient_seconds=completed - gradient_start,
            elapsed_seconds=completed - started,
            moving_grid=True,
            net_force_hartree_bohr=(fq.sum(0) + fm.sum(0)).tolist(),
        )
        np.save(directory / "quantum-forces-hartree-bohr.npy", fq)
        np.save(directory / "embedding-forces-hartree-bohr.npy", fm)
        _record(directory / "result.json", observation)
        return {
            "observation": observation,
            "density": density,
            "quantum_force_hartree_bohr": fq,
            "embedding_force_hartree_bohr": fm,
        }
    except Exception as exc:
        _record(
            directory / "failure.json",
            {"exception": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        if mol is not None:
            mol.stdout.close()
