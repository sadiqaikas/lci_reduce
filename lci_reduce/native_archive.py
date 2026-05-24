"""Offline conversion of native openLCA archives to JSON-LD ZIP."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from .errors import NativeArchiveConversionError


JAVA_SOURCE = "native_support/OpenLcaJsonExport.java"
ECJ_JAR = "vendor/ecj-3.45.0.jar"


@dataclass(frozen=True)
class OpenLcaRuntime:
    java_path: Path
    jar_paths: tuple[Path, ...]
    home: Path


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _source_file() -> Path:
    return _package_root() / JAVA_SOURCE


def _source_dir() -> Path:
    return _package_root() / "native_support"


def _ecj_jar() -> Path:
    return _package_root() / ECJ_JAR


def _cache_root() -> Path:
    path = Path(tempfile.gettempdir()) / "lci_reduce-java"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _conversion_cache_root() -> Path:
    path = _cache_root() / "converted"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_openlca_homes() -> list[Path]:
    candidates: list[Path] = []
    for key in ("OPENLCA_APP_HOME", "OPENLCA_HOME", "OLCA_APP_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path("/Applications/openLCA.app"),
            Path.home() / "Applications" / "openLCA.app",
            Path("C:/Program Files/openLCA"),
            Path("C:/Program Files/openLCA/openLCA.app"),
        ]
    )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(candidate)
    return ordered


def _jar_paths_from_plugins(plugins_dir: Path) -> tuple[Path, ...]:
    jar_paths = sorted(path for path in plugins_dir.rglob("*.jar") if path.is_file())
    return tuple(jar_paths)


def _runtime_from_home(home: Path) -> OpenLcaRuntime | None:
    if not home.exists():
        return None
    candidates = [
        home / "Contents" / "Eclipse",
        home,
    ]
    for eclipse_home in candidates:
        plugins_dir = eclipse_home / "plugins"
        java_path = eclipse_home / "jre" / "Contents" / "Home" / "bin" / "java"
        if not plugins_dir.is_dir() or not java_path.is_file():
            continue
        jar_paths = _jar_paths_from_plugins(plugins_dir)
        if not jar_paths:
            continue
        return OpenLcaRuntime(java_path=java_path, jar_paths=jar_paths, home=home)
    return None


def find_openlca_runtime() -> OpenLcaRuntime:
    for candidate in _candidate_openlca_homes():
        runtime = _runtime_from_home(candidate)
        if runtime is not None:
            return runtime
    raise NativeArchiveConversionError(
        "Native openLCA archive support requires a local openLCA installation with its bundled Java runtime. "
        "Set OPENLCA_APP_HOME if openLCA is installed in a non-standard location."
    )


def _classpath(paths: list[Path]) -> str:
    return os.pathsep.join(str(path) for path in paths)


def _helper_build_dir(runtime: OpenLcaRuntime) -> Path:
    source_dir = _source_dir()
    source_files = sorted(source_dir.glob("*.java"))
    digest = hashlib.sha256(
        (
            "|".join(str(path) for path in runtime.jar_paths)
            + "|"
            + "|".join(str(path) for path in source_files)
        ).encode("utf-8")
    ).hexdigest()[:16]
    build_dir = _cache_root() / f"build-{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _helper_class_file(build_dir: Path) -> Path:
    return build_dir / "OpenLcaJsonExport.class"


def _compile_helper(runtime: OpenLcaRuntime, build_dir: Path) -> None:
    ecj_jar = _ecj_jar()
    source_dir = _source_dir()
    source_files = sorted(source_dir.glob("*.java"))
    if not ecj_jar.is_file():
        raise NativeArchiveConversionError(f"Missing bundled Java compiler jar: {ecj_jar}")
    if not source_files:
        raise NativeArchiveConversionError(f"Missing Java helper sources in {source_dir}")
    classpath = _classpath(list(runtime.jar_paths))
    cmd = [
        str(runtime.java_path),
        "-jar",
        str(ecj_jar),
        "-proc:none",
        "-source",
        "21",
        "-target",
        "21",
        "-cp",
        classpath,
        "-d",
        str(build_dir),
        *[str(path) for path in source_files],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise NativeArchiveConversionError(f"Failed to compile native archive helper: {detail}")


def ensure_compiled_helper(runtime: OpenLcaRuntime) -> Path:
    build_dir = _helper_build_dir(runtime)
    class_file = _helper_class_file(build_dir)
    source_files = sorted(_source_dir().glob("*.java"))
    newest_source_mtime = max(path.stat().st_mtime for path in source_files)
    if class_file.exists() and class_file.stat().st_mtime >= newest_source_mtime:
        return build_dir
    _compile_helper(runtime, build_dir)
    if not class_file.exists():
        raise NativeArchiveConversionError(f"Java helper compilation did not produce {class_file}")
    return build_dir


def _run_helper_class(runtime: OpenLcaRuntime, class_name: str, args: list[str]) -> subprocess.CompletedProcess:
    build_dir = ensure_compiled_helper(runtime)
    run_classpath = _classpath([build_dir, *runtime.jar_paths])
    cmd = [
        str(runtime.java_path),
        "-cp",
        run_classpath,
        class_name,
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _extract_archive_to_tempdir(source: Path, temp_root: Path) -> Path:
    extracted = temp_root / "native-db"
    extracted.mkdir(parents=True, exist_ok=True)
    with ZipFile(source, "r") as archive:
        archive.extractall(extracted)
    return extracted


def convert_native_archive_to_jsonld(source: str, output_zip: str, *, scope: str = "full") -> str:
    source_path = Path(source)
    output_path = Path(output_zip)
    runtime = find_openlca_runtime()
    with tempfile.TemporaryDirectory(prefix="lci_reduce-native-") as temp_dir_name:
        temp_root = Path(temp_dir_name)
        database_dir = source_path if source_path.is_dir() else _extract_archive_to_tempdir(source_path, temp_root)
        result = _run_helper_class(runtime, "OpenLcaJsonExport", [str(database_dir), str(output_path), scope])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise NativeArchiveConversionError(f"Failed to convert native openLCA archive to JSON-LD: {detail}")
    if not output_path.is_file():
        raise NativeArchiveConversionError(f"Native archive conversion did not produce {output_path}")
    return str(output_path)


def convert_native_archive_to_temp_jsonld(source: str, *, scope: str = "full") -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="lci_reduce-jsonld-"))
    output_path = temp_dir / f"{Path(source).stem}_native_export.zip"
    try:
        return convert_native_archive_to_jsonld(source, str(output_path), scope=scope)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def get_cached_native_archive_jsonld(source: str, *, scope: str = "full") -> str:
    source_path = Path(source)
    source_stat = source_path.stat()
    helper_state = max(path.stat().st_mtime_ns for path in _source_dir().glob("*.java"))
    digest = hashlib.sha256(
        (
            str(source_path.resolve())
            + "|"
            + str(source_stat.st_size)
            + "|"
            + str(source_stat.st_mtime_ns)
            + "|"
            + scope
            + "|"
            + str(helper_state)
        ).encode("utf-8")
    ).hexdigest()[:24]
    output_path = _conversion_cache_root() / f"{source_path.stem}-{scope}-{digest}.zip"
    if output_path.is_file():
        return str(output_path)
    return convert_native_archive_to_jsonld(source, str(output_path), scope=scope)


def build_native_archive_from_jsonld(source_jsonld_zip: str, output_archive: str) -> str:
    runtime = find_openlca_runtime()
    output_path = Path(output_archive)
    with tempfile.TemporaryDirectory(prefix="lci_reduce-build-native-") as temp_dir_name:
        temp_root = Path(temp_dir_name)
        database_dir = temp_root / "native-db"
        result = _run_helper_class(runtime, "OpenLcaJsonImport", [source_jsonld_zip, str(database_dir)])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise NativeArchiveConversionError(f"Failed to build native openLCA archive from JSON-LD: {detail}")
        with ZipFile(output_path, "w") as archive:
            for file_path in sorted(database_dir.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(database_dir).as_posix())
    if not output_path.is_file():
        raise NativeArchiveConversionError(f"Native archive build did not produce {output_path}")
    return str(output_path)
