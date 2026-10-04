#!/usr/bin/env python3
"""Build reviewable Git-snapshot/source and local Windows portable releases.

Only Python's standard library is used. This tool never installs dependencies,
runs tests, launches the application, or accesses user workspaces.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parent.parent
CHUNK = 1024 * 1024
MANIFEST_NAME = "RELEASE-MANIFEST.json"
PAYLOAD_SUMS_NAME = "PAYLOAD-SHA256SUMS"
PRIVATE_ROOTS = {".git", ".codegraph", ".runtime", ".validation", ".launcher-install", "workspace", "workspaces", "release-output"}
PRIVATE_PARTS = {"__pycache__", ".pytest_cache", ".vs"}
SHA = re.compile(r"^[0-9a-f]{64}$")


class ReleaseError(Exception):
    pass


def git(*args: str, repository: Path = REPO) -> bytes:
    result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True)
    if result.returncode:
        raise ReleaseError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def relative(value: str) -> str:
    if "\\" in value or any(ord(char) < 32 for char in value) or ":" in value:
        raise ReleaseError(f"Unsafe relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in {".", ".."} for x in value.split("/")):
        raise ReleaseError(f"Unsafe relative path: {value!r}")
    return p.as_posix()


def local(value: str) -> Path:
    p = REPO.joinpath(*PurePosixPath(relative(value)).parts)
    try:
        p.resolve(strict=True).relative_to(REPO)
    except (ValueError, OSError) as exc:
        raise ReleaseError(f"Missing file or path outside repository: {value}") from exc
    return p


def matches(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value.casefold(), p.casefold()) for p in patterns)


def private(value: str) -> bool:
    parts = PurePosixPath(value).parts
    return parts[0].casefold() in PRIVATE_ROOTS or any(x.casefold() in PRIVATE_PARTS for x in parts)


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Entry:
    path: str
    size: int
    category: str
    source: Path | None = None
    blob: str | None = None
    data: bytes | None = None
    expected_sha256: str | None = None

    def content(self):
        if self.data is not None:
            return io.BytesIO(self.data)
        if self.blob:
            return io.BytesIO(git("cat-file", "blob", self.blob))
        if self.source is None:
            raise ReleaseError(f"Entry has no content: {self.path}")
        try:
            self.source.resolve(strict=True).relative_to(REPO)
        except (OSError, ValueError) as exc:
            raise ReleaseError(f"File moved outside repository: {self.path}") from exc
        return self.source.open("rb")


class Plan:
    def __init__(self, kind: str, config: dict, commit: str, weights: str):
        self.kind, self.config, self.commit, self.weights = kind, config, commit, weights
        self.entries: dict[str, Entry] = {}
        self.case_names: set[str] = set()
        self.excluded = 0
        self.packages: list[dict] = []
        self.warnings: list[str] = []

    def add(self, entry: Entry):
        name = relative(entry.path)
        if name in {MANIFEST_NAME, PAYLOAD_SUMS_NAME}:
            raise ReleaseError(f"Reserved release filename: {name}")
        if name in self.entries:
            raise ReleaseError(f"Duplicate planned file: {name}")
        if name.casefold() in self.case_names:
            raise ReleaseError(f"Windows filename collision: {name}")
        self.entries[name] = entry
        self.case_names.add(name.casefold())

    def file(self, source_name: str, target: str | None = None, category: str = "runtime", expected: str | None = None):
        p = local(source_name)
        if not p.is_file():
            raise ReleaseError(f"Required regular file missing: {source_name}")
        self.add(Entry(target or source_name, p.stat().st_size, category, source=p, expected_sha256=expected))

    def tree(self, source_name: str, target: str | None = None, category: str = "runtime", skip_modules: bool = False):
        source = local(source_name)
        if not source.is_dir():
            raise ReleaseError(f"Required directory missing: {source_name}")
        base_target = target or source_name

        def walk(directory: Path, output: str, ancestors: set[Path]):
            real = directory.resolve(strict=True)
            try:
                real.relative_to(REPO)
            except ValueError as exc:
                raise ReleaseError(f"Directory escapes repository: {output}") from exc
            if real in ancestors:
                raise ReleaseError(f"Directory link cycle: {output}")
            ancestors = ancestors | {real}
            for p in sorted(directory.iterdir(), key=lambda x: x.name.casefold()):
                dest = relative(output + "/" + p.name)
                if matches(dest, self.config["runtimeExclude"]) or matches(dest, self.config["portableExclude"]) or p.name.casefold() in PRIVATE_PARTS:
                    self.excluded += 1
                    continue
                if skip_modules and p.name == "node_modules":
                    continue
                if p.is_dir():
                    walk(p, dest, ancestors)
                elif p.is_file():
                    try:
                        p.resolve(strict=True).relative_to(REPO)
                    except ValueError as exc:
                        raise ReleaseError(f"File link escapes repository: {dest}") from exc
                    self.add(Entry(dest, p.stat().st_size, category, source=p))
                else:
                    raise ReleaseError(f"Unsupported filesystem entry: {dest}")
        walk(source, base_target, set())

    def npm(self):
        # Resolve the installed production graph, then materialize normal files.
        # No global module path, external hoisted tree, .pnpm store, dev dependency
        # or reparse point is placed in the resulting archive.
        root_manifest = json.loads(local("webui/package.json").read_text(encoding="utf-8"))
        base = "webui/node_modules"
        roots: dict[str, tuple[Path, str]] = {}
        queue: list[tuple[Path, str, list[tuple[str, Path, str]]]] = []
        destinations: set[str] = set()

        def resolve_package(start: Path, name: str) -> Path | None:
            if not re.fullmatch(r"(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+", name):
                raise ReleaseError(f"Unsafe npm package name: {name!r}")
            current = start.resolve()
            while current == REPO or REPO in current.parents:
                candidate = current / "node_modules" / name / "package.json"
                if candidate.is_file():
                    real = candidate.parent.resolve(strict=True)
                    try:
                        real.relative_to(REPO / "webui" / "node_modules")
                    except ValueError as exc:
                        raise ReleaseError(f"npm dependency uses an external location: {name}") from exc
                    return real
                if current == REPO:
                    break
                current = current.parent
            return None

        for name in sorted(root_manifest.get("dependencies", {})):
            source = resolve_package(REPO / "webui", name)
            if source is None:
                raise ReleaseError(f"Missing installed production dependency: {name}")
            dest = base + "/" + name
            roots[name] = (source, dest)
            queue.append((source, dest, [(name, source, dest)]))
        for name in sorted(root_manifest.get("optionalDependencies", {})):
            source = resolve_package(REPO / "webui", name)
            if source and name not in roots:
                dest = base + "/" + name
                roots[name] = (source, dest)
                queue.append((source, dest, [(name, source, dest)]))

        while queue:
            source, dest, ancestors = queue.pop(0)
            if dest in destinations:
                continue
            destinations.add(dest)
            package = json.loads((source / "package.json").read_text(encoding="utf-8"))
            self.tree(source.relative_to(REPO).as_posix(), dest, "javascript-dependency", skip_modules=True)
            self.packages.append({"name": package["name"], "version": package["version"], "path": dest, "license": package.get("license", "UNSPECIFIED")})
            required = set(package.get("dependencies", {}))
            optional = set(package.get("optionalDependencies", {}))
            peers = set(package.get("peerDependencies", {}))
            for name in sorted(required | optional | peers):
                child = resolve_package(source, name)
                if child is None:
                    if name in required and name not in optional:
                        raise ReleaseError(f"Missing dependency {name} required by {package['name']}")
                    if name in peers and not package.get("peerDependenciesMeta", {}).get(name, {}).get("optional", False):
                        raise ReleaseError(f"Missing peer dependency {name} required by {package['name']}")
                    continue
                if any(n == name and s == child for n, s, _ in ancestors):
                    continue
                if name not in roots:
                    child_dest = base + "/" + name
                    roots[name] = (child, child_dest)
                elif roots[name][0] == child:
                    child_dest = roots[name][1]
                else:
                    child_dest = dest + "/node_modules/" + name
                queue.append((child, child_dest, ancestors + [(name, child, child_dest)]))

    def summary(self) -> dict:
        groups: dict[str, dict] = {}
        for e in self.entries.values():
            group = groups.setdefault(e.category, {"files": 0, "bytes": 0})
            group["files"] += 1
            group["bytes"] += e.size
        return {"kind": self.kind, "gitCommit": self.commit, "weights": self.weights if self.kind == "portable" else None,
                "fileCount": len(self.entries), "totalBytes": sum(e.size for e in self.entries.values()),
                "excludedEntries": self.excluded, "groups": groups, "warnings": self.warnings,
                "npmPackages": self.packages}


def plan_release(kind: str, config: dict, commit: str, weights: str) -> Plan:
    plan = Plan(kind, config, commit, weights)
    for record in git("ls-tree", "-rlz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        meta, raw_name = record.split(b"\t", 1)
        mode, kind_name, oid, size = meta.decode("ascii").split()
        name = relative(raw_name.decode("utf-8"))
        if private(name) or matches(name, config["sourceExclude"]) or (kind == "portable" and matches(name, config["portableExclude"])):
            plan.excluded += 1
            continue
        if kind_name != "blob" or mode not in {"100644", "100755"}:
            raise ReleaseError(f"Git links/submodules require an explicit distribution policy: {name}")
        plan.add(Entry(name, int(size), "source", blob=oid))
    # Generic XSeg is an inference asset required in both source and portable
    # archives. It must never disappear behind the optional helper-weight flag.
    generic_bytes = git("show", commit + ":" + config["genericXSegManifest"])
    generic = json.loads(generic_bytes)
    if generic.get("schemaVersion") != 1 or generic.get("licenseStatus") != "verified":
        raise ReleaseError("Generic XSeg source/license review is incomplete")
    plan.file(generic["converted"]["path"], category="generic-xseg-weight", expected=generic["converted"]["sha256"])
    plan.file(generic["metadata"]["source"], generic["metadata"]["path"], "generic-xseg-metadata", expected=generic["metadata"]["sha256"])
    plan.add(Entry("_internal/model_generic_xseg/SOURCE.json", len(generic_bytes), "generic-xseg-source", data=generic_bytes))
    for key, category in (("license", "third-party-license"), ("summary", "generic-xseg-source")):
        item = generic[key]
        plan.file(item["source"], item["path"], category, expected=item["sha256"])
    if kind == "source":
        return plan
    for name in config["runtimeTrees"]:
        plan.tree(name, category="built-webui" if name == "webui/dist" else "runtime")
    for name in config["runtimeFiles"]:
        if name in plan.entries:
            continue
        expected = config.get("sfaceSha256") if name.endswith("face_recognition_sface_2021dec.onnx") else None
        plan.file(name, category="launcher" if name.startswith("launcher/bin/") else "runtime", expected=expected)
    for item in config["licenseCopies"]:
        plan.file(item["source"], item["target"], "third-party-license")
    # setup-runtime.ps1 replaces this with the current absolute base path before
    # application startup. Never export the workstation's absolute pyvenv paths.
    old_cfg = local(".venv/pyvenv.cfg").read_text(encoding="utf-8")
    version = re.search(r"^version\s*=\s*(.+)$", old_cfg, re.MULTILINE)
    if not version:
        raise ReleaseError("Cannot determine the local venv Python version")
    cfg = ("home = ../_internal/python_base\ninclude-system-site-packages = false\nversion = " + version.group(1).strip() + "\n").encode("utf-8")
    plan.add(Entry(".venv/pyvenv.cfg", len(cfg), "portable-configuration", data=cfg))
    if weights == "bundled":
        for item in config["auxiliaryWeights"]:
            plan.file(item["path"], category="auxiliary-weight", expected=item["sha256"])
            if item.get("licenseStatus") != "verified":
                plan.warnings.append("Bundling awaits a documented upstream source/license review: " + item["path"])
    else:
        plan.warnings.append("Four auxiliary weights are omitted; run tools/prepare-vision-runtime.ps1 before using extraction/enhancement.")
    notes = ("DFL-PT-WEBUI portable runtime\n"
             "Start with the project launcher or 启动 WebUI.bat; startup repairs pyvenv.cfg for the current directory.\n"
             "Microsoft Edge WebView2 Runtime and an appropriate NVIDIA driver are host prerequisites.\n"
             "Auxiliary weight policy: " + weights + ".\n"
             + ("Obtain auxiliary weights: powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\prepare-vision-runtime.ps1\n" if weights == "download" else "")
             + "Verified generic XSeg inference weights are included under _internal/model_generic_xseg in both archive kinds.\n"
             "Packaging and file integrity checks do not constitute functional, training, dual-GPU or visual-quality validation.\n").encode("utf-8")
    plan.add(Entry("RELEASE-RUNTIME-NOTES.txt", len(notes), "release-note", data=notes))
    plan.npm()
    required = ["_internal/python_base/python.exe", "_internal/python_base/python312.dll", "_internal/python_base/LICENSE.txt",
                ".venv/Scripts/python.exe", ".venv/Lib/site-packages/torch/__init__.py", ".venv/Lib/site-packages/torch/lib/torch_cpu.dll",
                "_internal/node/bin/node.exe", "_internal/node/bin/npm.cmd", "_internal/node/bin/LICENSE",
                "_internal/ffmpeg/ffmpeg.exe", "_internal/ffmpeg/ffprobe.exe", "_internal/ffmpeg/LICENSE",
                "webui/dist/client/index.html", "webui/dist/.build-provenance.json", "webui/node_modules/ws/package.json",
                "webui/node_modules/vite/bin/vite.js", "webui/node_modules/node-pty/package.json",
                "launcher/bin/DFL-PT-WEBUI.Launcher.exe", "tools/licenses/NotoFonts-OFL-1.1.txt",
                "_internal/model_generic_xseg/XSeg_256.pth", "_internal/model_generic_xseg/XSeg_data.dat",
                "_internal/model_generic_xseg/SOURCE.json", "_internal/model_generic_xseg/LICENSE.txt"]
    for name in required:
        if name not in plan.entries:
            raise ReleaseError(f"Portable inventory omitted a required artifact: {name}")
    native_prefixes = ("webui/node_modules/node-pty/build/Release/", "webui/node_modules/node-pty/prebuilds/win32-x64/")
    for module in ("conpty.node", "conpty_console_list.node", "pty.node"):
        if not any(prefix + module in plan.entries for prefix in native_prefixes):
            raise ReleaseError(f"Windows x64 node-pty native module missing: {module}")
    return plan


def zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 0
    info.external_attr = 0o100644 << 16
    info._compresslevel = 1
    return info


def write_new(path: Path, content: bytes):
    with path.open("xb") as f:
        f.write(content)


def build(plan: Plan, output: Path, version: dict, config_sha: str, max_part_bytes: int, compression: int, source_commit: str, source_tree: str) -> dict:
    if plan.kind == "portable" and plan.weights == "bundled" and plan.warnings:
        raise ReleaseError("\n".join(plan.warnings) + "\nResolve release/package-config.json licenseStatus with evidence, or use --weights download.")
    name = f"{version['product']}-{version['version']}-{plan.kind}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ReleaseError("Unsafe product/version release filename")
    root = version["product"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", root):
        raise ReleaseError("Unsafe product archive root")
    archive = output / (name + ".zip")
    manifest_path = output / (name + ".manifest.json")
    index_path = output / (name + ".archive.json")
    if any(output.glob(name + ".*")):
        raise ReleaseError(f"Release files already exist for {name}; use a fresh output directory")
    records = []
    try:
        with zipfile.ZipFile(archive, "x", compression=compression, compresslevel=1, allowZip64=True) as z:
            for entry in sorted(plan.entries.values(), key=lambda x: x.path):
                h = hashlib.sha256()
                count = 0
                before = entry.source.stat() if entry.source else None
                with entry.content() as source, z.open(zip_info(root + "/" + entry.path, compression), "w", force_zip64=entry.size >= 2 * 1024**3) as target:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        target.write(block)
                        h.update(block)
                        count += len(block)
                sha = h.hexdigest()
                if count != entry.size:
                    raise ReleaseError(f"File size changed while archiving: {entry.path}")
                if before:
                    after = entry.source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ReleaseError(f"File changed while archiving: {entry.path}")
                if entry.expected_sha256 and sha != entry.expected_sha256:
                    raise ReleaseError(f"Pinned SHA-256 mismatch: {entry.path}")
                records.append({"path": entry.path, "size": count, "sha256": sha, "category": entry.category})
            manifest = {"schemaVersion": 1, "product": version["product"], "version": version["version"], "kind": plan.kind,
                        "sourceCommit": source_commit, "sourceTree": source_tree, "archiveRoot": root, "createdUtc": datetime.now(timezone.utc).isoformat(),
                        "configSha256": config_sha, "weights": plan.weights if plan.kind == "portable" else None,
                        "fileCount": len(records), "totalBytes": sum(r["size"] for r in records),
                        "npmPackages": plan.packages, "warnings": plan.warnings, "files": records}
            manifest_bytes = encoded(manifest)
            sums = "".join(f"{r['sha256']}  {r['path']}\n" for r in records).encode("utf-8")
            z.writestr(zip_info(root + "/" + MANIFEST_NAME, compression), manifest_bytes)
            z.writestr(zip_info(root + "/" + PAYLOAD_SUMS_NAME, compression), sums)
        archive_size, archive_sha = archive.stat().st_size, digest_file(archive)
        parts = []
        if archive_size > max_part_bytes:
            with archive.open("rb") as source:
                number = 1
                while source.tell() < archive_size:
                    part = output / (archive.name + f".part{number:03d}")
                    h, size = hashlib.sha256(), 0
                    with part.open("xb") as target:
                        while size < max_part_bytes:
                            block = source.read(min(CHUNK, max_part_bytes - size))
                            if not block:
                                break
                            target.write(block)
                            h.update(block)
                            size += len(block)
                    parts.append({"file": part.name, "size": size, "sha256": h.hexdigest()})
                    number += 1
            archive.unlink()  # This exact newly-created file belongs to this build.
        else:
            parts.append({"file": archive.name, "size": archive_size, "sha256": archive_sha})
        write_new(manifest_path, manifest_bytes)
        index = {"schemaVersion": 1, "archiveFile": archive.name, "archiveSize": archive_size, "archiveSha256": archive_sha,
                 "manifestFile": manifest_path.name, "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                 "parts": parts}
        write_new(index_path, encoded(index))
        print(f"Built {plan.kind}: {len(records):,} payload files, {archive_size:,} archive bytes, {len(parts)} part(s)", flush=True)
        return index
    except Exception:
        # Keep partial output for diagnosis; never remove user-owned directories.
        print(f"Build failed; partial files may remain under {output}", file=sys.stderr)
        raise


class PartsReader(io.RawIOBase):
    """A seekable, read-only ZIP view over ordinary numbered binary parts."""
    def __init__(self, paths: list[Path]):
        super().__init__()
        self.handles = [p.open("rb") for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.position = 0
        self.total = sum(self.sizes)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position + offset if whence == 1 else self.total + offset
        if pos < 0:
            raise ValueError("negative seek")
        self.position = pos
        return pos

    def read(self, size=-1):
        remaining = max(0, self.total - self.position) if size < 0 else min(size, max(0, self.total - self.position))
        blocks = []
        base = 0
        for handle, length in zip(self.handles, self.sizes):
            if remaining and base <= self.position < base + length:
                handle.seek(self.position - base)
                block = handle.read(min(remaining, base + length - self.position))
                if not block:
                    raise ReleaseError("Unexpected end of archive part")
                blocks.append(block)
                self.position += len(block)
                remaining -= len(block)
            base += length
        return b"".join(blocks)

    def close(self):
        for handle in self.handles:
            handle.close()
        super().close()


def artifact_path(directory: Path, name: str) -> Path:
    if relative(name) != PurePosixPath(name).name:
        raise ReleaseError(f"Artifact index must reference a basename: {name!r}")
    p = directory / name
    try:
        p.resolve(strict=True).relative_to(directory.resolve())
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"Missing or external release artifact: {name}") from exc
    if not p.is_file():
        raise ReleaseError(f"Missing release artifact: {name}")
    return p


def verify(index_path: Path, sums_path: Path | None):
    directory = index_path.resolve().parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schemaVersion") != 1 or not index.get("parts"):
        raise ReleaseError("Unsupported or empty archive index")
    paths = []
    joined = hashlib.sha256()
    for part in index["parts"]:
        p = artifact_path(directory, part["file"])
        if not SHA.fullmatch(part["sha256"]) or p.stat().st_size != part["size"]:
            raise ReleaseError(f"Part size/hash metadata is invalid: {p.name}")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
                joined.update(block)
        if h.hexdigest() != part["sha256"]:
            raise ReleaseError(f"Part SHA-256 mismatch: {p.name}")
        paths.append(p)
    if len({p.name.casefold() for p in paths}) != len(paths) or sum(p.stat().st_size for p in paths) != index["archiveSize"] or joined.hexdigest() != index["archiveSha256"]:
        raise ReleaseError("Combined archive size/SHA-256 mismatch")
    manifest_path = artifact_path(directory, index["manifestFile"])
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != index["manifestSha256"]:
        raise ReleaseError("External manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schemaVersion") != 1:
        raise ReleaseError("Unsupported file manifest")
    root = relative(manifest["archiveRoot"])
    if "/" in root:
        raise ReleaseError("Archive root must be a single directory")
    records = manifest["files"]
    expected = {root + "/" + relative(r["path"]): r for r in records}
    if len(expected) != len(records) or len({n.casefold() for n in expected}) != len(expected):
        raise ReleaseError("Duplicate manifest file path")
    expected_names = set(expected) | {root + "/" + MANIFEST_NAME, root + "/" + PAYLOAD_SUMS_NAME}
    with PartsReader(paths) as reader, zipfile.ZipFile(reader, "r") as z:
        infos = z.infolist()
        if len(infos) != len(expected_names) or {i.filename for i in infos} != expected_names:
            raise ReleaseError("Archive entries do not exactly match the manifest")
        if z.read(root + "/" + MANIFEST_NAME) != manifest_bytes:
            raise ReleaseError("Embedded/external manifests differ")
        required_sums = "".join(f"{r['sha256']}  {r['path']}\n" for r in records).encode("utf-8")
        if z.read(root + "/" + PAYLOAD_SUMS_NAME) != required_sums:
            raise ReleaseError("Embedded payload checksum list differs from the manifest")
        for info in infos:
            if info.filename not in expected:
                continue
            record = expected[info.filename]
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000 or info.file_size != record["size"] or not SHA.fullmatch(record["sha256"]):
                raise ReleaseError(f"Invalid file entry: {info.filename}")
            h = hashlib.sha256()
            with z.open(info, "r") as f:
                for block in iter(lambda: f.read(CHUNK), b""):
                    h.update(block)
            if h.hexdigest() != record["sha256"]:
                raise ReleaseError(f"Payload SHA-256 mismatch: {info.filename}")
    if manifest["fileCount"] != len(records) or manifest["totalBytes"] != sum(r["size"] for r in records):
        raise ReleaseError("Manifest totals are inconsistent")
    if sums_path:
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            sha, name = line.split("  ", 1)
            if not SHA.fullmatch(sha) or digest_file(artifact_path(sums_path.resolve().parent, name)) != sha:
                raise ReleaseError(f"Published artifact SHA-256 mismatch: {name}")
    print(f"Verified read-only: {manifest['kind']}, {len(records):,} files, {manifest['totalBytes']:,} payload bytes; all parts, paths, sizes and SHA-256 values match.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "build"):
        p = sub.add_parser(command)
        p.add_argument("--kind", choices=("source", "portable", "both"), default="source")
        p.add_argument("--weights", choices=("bundled", "download"), default="bundled")
        p.add_argument("--ref", default="HEAD", help="Committed snapshot; builds require a clean checkout at this commit")
        p.add_argument("--source-commit", help="Public snapshot commit to record instead of the local development commit")
        p.add_argument("--source-repository", type=Path, help="Read-only repository containing --source-commit; its tree must match --ref")
        p.add_argument("--config", type=Path, default=REPO / "release/package-config.json")
        p.add_argument("--output-dir", type=Path, default=REPO / "release-output")
        if command == "preview":
            p.add_argument("--write-plan", action="store_true", help="Write a path/size plan without hashing or archiving")
            p.add_argument("--force-plan", action="store_true", help="Explicitly allow replacing the two preview plan files")
        else:
            p.add_argument("--max-part-mib", type=int, default=1900, help="Maximum artifact part size; range 1..1900 MiB")
            p.add_argument("--compression", choices=("stored", "deflated"), default="deflated")
    p = sub.add_parser("verify", help="Read-only verification; never extracts or joins parts")
    p.add_argument("--archive-index", type=Path, required=True)
    p.add_argument("--sha256sums", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.archive_index, args.sha256sums)
        return
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schemaVersion") != 1:
        raise ReleaseError("Unsupported package config schema")
    commit = git("rev-parse", "--verify", args.ref + "^{commit}").decode("ascii").strip()
    source_tree = git("rev-parse", commit + "^{tree}").decode("ascii").strip()
    if args.source_repository and not args.source_commit:
        raise ReleaseError("--source-repository requires --source-commit")
    source_commit = commit
    if args.source_commit:
        public_repository = args.source_repository.resolve() if args.source_repository else REPO
        source_commit = git("rev-parse", "--verify", args.source_commit + "^{commit}", repository=public_repository).decode("ascii").strip()
        public_tree = git("rev-parse", source_commit + "^{tree}", repository=public_repository).decode("ascii").strip()
        if public_tree != source_tree:
            raise ReleaseError("Public snapshot and --ref code tree differ; rebuild the snapshot before packaging")
    version = json.loads(git("show", commit + ":release/version.json"))
    kinds = ["source", "portable"] if args.kind == "both" else [args.kind]
    if args.command == "build":
        if not 1 <= args.max_part_mib <= 1900:
            raise ReleaseError("--max-part-mib must be 1..1900 (below the GitHub 2 GiB asset ceiling)")
        if git("status", "--porcelain=v1", "--untracked-files=all").strip():
            raise ReleaseError("Release build requires a clean Git worktree, including untracked files")
        if commit != git("rev-parse", "HEAD").decode("ascii").strip():
            raise ReleaseError("Portable artifacts must match the checked-out commit; check out --ref before building")
        if "portable" in kinds:
            node = local("_internal/node/bin/node.exe")
            result = subprocess.run([str(node), str(local("tools/dist-provenance.mjs")), "check"], cwd=REPO, capture_output=True, text=True)
            if result.returncode:
                raise ReleaseError("Read-only WebUI build freshness check failed: " + result.stderr.strip())
    plans = [plan_release(kind, config, commit, args.weights) for kind in kinds]
    output = args.output_dir.resolve()
    # Outputs may live in the dedicated ignored folder or outside the checkout.
    if output == REPO or (REPO in output.parents and (REPO / "release-output") not in (output, *output.parents)):
        raise ReleaseError("Output must be inside release-output/ or outside the repository")
    for plan in plans:
        print(json.dumps(plan.summary(), ensure_ascii=False, indent=2), flush=True)
    if args.command == "preview":
        if args.force_plan and not args.write_plan:
            raise ReleaseError("--force-plan requires --write-plan")
        if args.write_plan:
            output.mkdir(parents=True, exist_ok=True)
            targets = [output / (plan.kind + ".plan.json") for plan in plans]
            if not args.force_plan and any(target.exists() for target in targets):
                raise ReleaseError("Preview plan exists; use a new output directory or explicitly pass --force-plan")
            for plan in plans:
                payload = {**plan.summary(), "files": [{"path": e.path, "size": e.size, "category": e.category} for e in sorted(plan.entries.values(), key=lambda x: x.path)]}
                target = output / (plan.kind + ".plan.json")
                if args.force_plan:
                    if target.is_symlink() or (target.exists() and not target.is_file()):
                        raise ReleaseError(f"Preview plan target is not an ordinary file: {target}")
                    target.write_bytes(encoded(payload))
                else:
                    write_new(target, encoded(payload))
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SHA256SUMS").exists() or (output / "restore-release.ps1").exists():
        raise ReleaseError("Output already has a checksum/restore file; use a fresh output directory")
    generic_entries = [entry for entry in plans[0].entries.values() if entry.category == "generic-xseg-weight"]
    if len(generic_entries) != 1:
        raise ReleaseError("Release must have exactly one pinned generic XSeg inference weight")
    generic_entry = generic_entries[0]
    generic_asset = output / PurePosixPath(generic_entry.path).name
    if generic_asset.exists():
        raise ReleaseError("Standalone XSeg asset already exists; use a fresh output directory")
    if shutil.disk_usage(output).free < 2 * sum(p.summary()["totalBytes"] for p in plans):
        raise ReleaseError("Insufficient free space for archive creation and part splitting (reserve twice the payload size)")
    indices = [build(plan, output, version, hashlib.sha256(config_bytes).hexdigest(), args.max_part_mib * 1024**2,
                     zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED, source_commit, source_tree) for plan in plans]
    # Git-clone installation fetches the same byte-identical inference model.
    # Archive users already receive this asset inside either package.
    with generic_entry.content() as source, generic_asset.open("xb") as target:
        shutil.copyfileobj(source, target, CHUNK)
    if generic_asset.stat().st_size != generic_entry.size or digest_file(generic_asset) != generic_entry.expected_sha256:
        raise ReleaseError("Standalone XSeg asset differs from the pinned archive weight")
    write_new(output / "restore-release.ps1", local("tools/restore-release.ps1").read_bytes())
    names = {"restore-release.ps1", generic_asset.name}
    for index in indices:
        names.add(index["manifestFile"])
        names.add(index["archiveFile"].removesuffix(".zip") + ".archive.json")
        names.update(p["file"] for p in index["parts"])
    write_new(output / "SHA256SUMS", "".join(f"{digest_file(output / name)}  {name}\n" for name in sorted(names)).encode("utf-8"))
    print(f"Release artifacts and SHA256SUMS: {output}")
    for index in indices:
        index_name = index["archiveFile"].removesuffix(".zip") + ".archive.json"
        print(f"Restore: powershell -NoProfile -ExecutionPolicy Bypass -File .\\restore-release.ps1 -ArchiveIndex .\\{index_name} -Destination .\\unpacked")


if __name__ == "__main__":
    try:
        main()
    except (ReleaseError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"release error: {exc}", file=sys.stderr)
        sys.exit(1)
