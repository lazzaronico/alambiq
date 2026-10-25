"""Build the measured Linux ARM64/CUDA 13 QM/MM extension in a selected environment.

Create the base environment from linux-aarch64-cuda13.explicit.txt first.
Build sources, logs, dependency revisions and installed-wheel receipts remain
in a fresh directory chosen by the researcher. Existing GPU4PySCF installations
retain their original files; select a fresh base environment for this recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CUPY_WHEEL = "cupy_cuda13x-14.0.1-cp312-cp312-manylinux2014_aarch64.whl"
SOURCES = {
    "gpu4pyscf-1.8.1.tar.gz": (
        "https://codeload.github.com/pyscf/gpu4pyscf/tar.gz/refs/tags/v1.8.1",
        "eb75553ed3e05c907511e6260407e279c8c2151d61bc74f7bfbaee2e6c422121",
    ),
    CUPY_WHEEL: (
        "https://files.pythonhosted.org/packages/7d/9b/7983b4e24749937dd4ab34565561a8c015e88df4ff9fd4678337b710b3ee/"
        + CUPY_WHEEL,
        "763f33f5641a28db5ac91788301271930e910813f1b1279119b504babeb1b863",
    ),
    "cuda_pathfinder-1.8.2-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/98/59/239c7259e669c46ddbcac0aa60e3a0ef00bfeaaa687f905b24dd6a7a10fe/cuda_pathfinder-1.8.2-py3-none-any.whl",
        "4e65059febdb4d19d5cbc4798677e19db2b582f2f702f457b609e571690d357e",
    ),
}
DEPENDENCIES = {
    "cutlass": "751eb9a8859ac36bfc77551f9e4a957c31a5a8b1",
    "libxc": "b225c254c063e1de835a4425115c9a6377478b32",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: object) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def verify_file(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Source SHA-256 differs; retained bytes: {path}")


def preflight(prefix: Path, work: Path, cuda: Path, compiler: Path) -> None:
    if (platform.system(), platform.machine()) != ("Linux", "aarch64"):
        raise ValueError("This measured recipe requires Linux ARM64")
    if work.exists() or any(
        work.is_relative_to(p) or p.is_relative_to(work) for p in (prefix, cuda)
    ):
        raise ValueError(
            "Choose a fresh build directory separate from the environment and CUDA toolkit"
        )
    if any(character.isspace() for character in str(work)):
        raise ValueError("Upstream native build paths require whitespace-free names")
    for path in (prefix / "bin/python", cuda / "bin/nvcc", compiler):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"An executable build prerequisite is required: {path}")


def build_environment(
    prefix: Path, work: Path, cuda: Path, compiler: Path
) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
        env.pop(name, None)
    env.update(
        PATH=f"{prefix}/bin:{cuda}/bin:/usr/bin:/bin",
        LD_LIBRARY_PATH=f"{prefix}/lib:{cuda}/lib64",
        CUDA_PATH=str(cuda),
        CUDACXX=str(cuda / "bin/nvcc"),
        CC="/usr/bin/gcc",
        CXX="/usr/bin/g++",
        CUDAHOSTCXX="/usr/bin/g++",
        FC=str(compiler),
        CONTRACT_ENGINE="cupy",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        PYTHONNOUSERSITE="1",
        TMPDIR=str(work / "tmp"),
        PIP_NO_CLEAN="1",
        CMAKE_BUILD_ARGS="-j 4",
        CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=ON -DCUDA_ARCHITECTURES=121 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_BUILD_TYPE=Release",
    )
    return env


def run(work: Path, name: str, command: list[str], env: dict[str, str]) -> None:
    dump(
        work / f"{name}.request.json",
        {"command": command, "started_at": datetime.now(UTC).isoformat()},
    )
    with (work / f"{name}.log").open("x") as stream:
        result = subprocess.run(
            command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    dump(
        work / f"{name}.json",
        {"returncode": result.returncode, "log_sha256": sha(work / f"{name}.log")},
    )
    result.check_returncode()
    print(f"Completed {name}; log: {work / (name + '.log')}", flush=True)


def fetch(work: Path, name: str) -> Path:
    url, expected = SOURCES[name]
    path = work / name
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        path.open("xb") as stream,
    ):
        headers, final_url = dict(response.headers), response.geturl()
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    dump(
        work / (name + ".source.json"),
        {
            "url": url,
            "final_url": final_url,
            "headers": headers,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        },
    )
    verify_file(path, expected)
    return path


def verify_dependencies(build: Path) -> dict[str, str]:
    paths = {
        "cutlass": build / "cutlass-prefix/src/cutlass",
        "libxc": build / "deps/src/libxc",
    }
    result = {}
    for name, path in paths.items():
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != DEPENDENCIES[name]:
            raise ValueError(f"Native dependency commit differs: {name}")
        if list(path.rglob("*.rej")):
            raise ValueError(f"Native dependency patch requires inspection: {path}")
        result[name] = commit
    return result


def install(prefix: Path, work: Path, cuda: Path, compiler: Path) -> None:
    prefix, work, cuda, compiler = (p.resolve() for p in (prefix, work, cuda, compiler))
    preflight(prefix, work, cuda, compiler)
    env = build_environment(prefix, work, cuda, compiler)
    for executable in ("/usr/bin/gcc", "/usr/bin/g++", "cmake", "git", "make"):
        if shutil.which(executable, path=env["PATH"]) is None:
            raise ValueError(f"Install this build prerequisite first: {executable}")
    work.mkdir(parents=True, exist_ok=False)
    (work / "tmp").mkdir()
    (work / "wheels").mkdir()
    python = str(prefix / "bin/python")
    dump(
        work / "intent.json",
        {
            "recipe_sha256": sha(Path(__file__)),
            "prefix": str(prefix),
            "cuda": str(cuda),
            "fortran_compiler": str(compiler),
            "fortran_compiler_sha256": sha(compiler),
            "sources": SOURCES,
            "dependencies": DEPENDENCIES,
            "build_flags": env["CMAKE_CONFIGURE_ARGS"],
            "build_parallelism": 4,
        },
    )
    probe = (
        "import importlib.util,sys,openmm,pyscf; "
        "assert sys.version_info[:2]==(3,12), 'Python 3.12 required'; "
        "assert openmm.__version__=='8.6' and pyscf.__version__=='2.14.0', 'Use the native base lock'; "
        "assert importlib.util.find_spec('gpu4pyscf') is None, 'Select a base environment with a fresh GPU4PySCF installation path'"
    )
    run(work, "base-check", [python, "-I", "-c", probe], env)
    run(work, "cuda-version", [str(cuda / "bin/nvcc"), "--version"], env)
    if "release 13.0," not in (work / "cuda-version.log").read_text():
        raise ValueError("The measured build requires CUDA toolkit 13.0")
    for name in SOURCES:
        fetch(work, name)
    with tarfile.open(work / "gpu4pyscf-1.8.1.tar.gz") as archive:
        archive.extractall(work, filter="data")
    source = work / "gpu4pyscf-1.8.1"
    run(
        work,
        "build-wheel",
        [
            python,
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--no-clean",
            "--wheel-dir",
            str(work / "wheels"),
            str(source),
        ],
        env,
    )
    builds = list((source / "build").glob("temp.*/gpu4pyscf"))
    if len(builds) != 1:
        raise ValueError("One retained native build tree is required")
    dependencies = verify_dependencies(builds[0])
    wheels = list((work / "wheels").glob("gpu4pyscf_cuda13x-1.8.1-*.whl"))
    if len(wheels) != 1:
        raise ValueError("One GPU4PySCF 1.8.1 CUDA 13 wheel is required")
    for name, (_, expected) in SOURCES.items():
        verify_file(work / name, expected)
    run(
        work,
        "install",
        [
            python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--report",
            str(work / "install-report.json"),
            str(work / CUPY_WHEEL),
            str(work / "cuda_pathfinder-1.8.2-py3-none-any.whl"),
            str(wheels[0]),
        ],
        env,
    )
    run(
        work,
        "isolated-import",
        [
            python,
            "-I",
            "-c",
            "import json,openmm,pyscf,cupy,gpu4pyscf; from gpu4pyscf.lib import cutensor; "
            "assert cupy.__version__=='14.0.1' and gpu4pyscf.__version__=='1.8.1'; "
            "assert cupy.cuda.Device().compute_capability=='121'; "
            "assert cutensor.contract_engine=='cupy'; "
            "assert float(cupy.arange(10,dtype=cupy.float64).sum().get())==45.; "
            "print(json.dumps({'openmm':openmm.__version__,'pyscf':pyscf.__version__,'cupy':cupy.__version__,"
            "'gpu4pyscf':gpu4pyscf.__version__,'module':gpu4pyscf.__file__,'device':cupy.cuda.Device().compute_capability}))",
        ],
        env,
    )
    dump(
        work / "installed.json",
        {
            "intent_sha256": sha(work / "intent.json"),
            "wheel_sha256": sha(wheels[0]),
            "dependencies": dependencies,
            "prefix": str(prefix),
            "records_sha256": {
                p.name: sha(p) for p in sorted(work.iterdir()) if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--fortran", type=Path, required=True)
    args = parser.parse_args()
    install(args.prefix, args.work, args.cuda, args.fortran)
