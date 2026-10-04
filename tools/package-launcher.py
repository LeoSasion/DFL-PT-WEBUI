"""Verify and package the independent PT launcher; standard library only.

Never starts the launcher, installs a runtime, or publishes a release. The
Windows PowerShell inspection reads assembly metadata and embedded resources
in reflection-only mode, so the launcher's startup code is not executed.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/LeoSasion/DFL-PT-WEBUI"
PRODUCT = "DFL-PT-WEBUI Launcher"
FILE_NAME = "DFL-PT-WEBUI.Launcher.exe"
MANIFEST_RESOURCE = "DflPtWebUi.Launcher.Payload.Manifest"
RESOURCE_PREFIX = "DflPtWebUi.Launcher.Payload.File."
FIXED_PAYLOAD = {
    "bootstrap/bootstrap.ps1": "launcher/bootstrap.ps1",
    "bootstrap/runtime-manifest.json": "launcher/runtime-manifest.json",
    "bootstrap/setup-runtime.ps1": "launcher/setup-runtime.ps1",
    "bootstrap/install-project.ps1": "launcher/install-project.ps1",
    "bootstrap/install-source.ps1": "launcher/install-source.ps1",
    "terminal/index.mjs": "launcher/server/index.mjs",
    "terminal/terminal-bridge.mjs": "launcher/server/terminal-bridge.mjs",
}
UI_RUNTIME_PACKAGES = {
    "react": "19.2.0", "react-dom": "19.2.0", "scheduler": "0.27.0",
    "@tabler/icons-react": "3.35.0", "@xterm/xterm": "6.0.0", "@xterm/addon-fit": "0.11.0",
}


class PackageError(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise PackageError("Required Git source identity check failed.")
    return result.stdout.strip()


def verified_source(root: Path, revision: str) -> dict:
    """Accept a clean matching tree on the existing sanitized public lineage."""
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise PackageError("Commit the reviewed source changes before packaging.")
    remote = git(root, "remote", "get-url", "public").rstrip("/")
    if remote.removesuffix(".git").casefold() != REPOSITORY.casefold():
        raise PackageError("The public remote must be the official PT repository.")
    commit = git(root, "rev-parse", "--verify", revision + "^{commit}")
    base = git(root, "rev-parse", "--verify", "refs/remotes/public/main^{commit}")
    tree = git(root, "rev-parse", commit + "^{tree}")
    if tree != git(root, "rev-parse", "HEAD^{tree}"):
        raise PackageError("The public source revision and tested checkout trees differ.")
    # A local private-history HEAD may have the same tree as a public snapshot.
    # Require the selected revision to continue the safe public history, without
    # merging any unrelated ancestry into the release's source identity.
    git(root, "merge-base", "--is-ancestor", base, commit)
    if git(root, "rev-list", "--min-parents=2", base + ".." + commit):
        raise PackageError("Public source revision merges an unrelated history.")
    return {"repository": REPOSITORY, "commit": commit, "tree": tree}


def verify_pe(path: Path) -> None:
    if not path.is_file() or not 1024 <= path.stat().st_size <= 64 * 1024 * 1024:
        raise PackageError("Launcher is not a bounded Windows executable.")
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise PackageError("Launcher is not a bounded Windows executable.")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset < 64 or offset + 26 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise PackageError("Launcher has an invalid PE header.")
    machine = struct.unpack_from("<H", data, offset + 4)[0]
    optional_magic = struct.unpack_from("<H", data, offset + 24)[0]
    if machine != 0x8664 or optional_magic != 0x20B:
        raise PackageError("Launcher must be Windows AMD64 / PE32+.")


INSPECT_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$path = $env:DFL_PT_PACKAGE_EXE
$assembly = [Reflection.Assembly]::ReflectionOnlyLoadFrom($path)
$attributes = @{}
foreach ($attribute in $assembly.GetCustomAttributesData()) {
    if ($attribute.AttributeType.Name -in @('AssemblyProductAttribute','AssemblyFileVersionAttribute')) {
        $attributes[$attribute.AttributeType.Name] = [string]$attribute.ConstructorArguments[0].Value
    }
}
$stream = $assembly.GetManifestResourceStream('DflPtWebUi.Launcher.Payload.Manifest')
if ($null -eq $stream) { throw 'Embedded payload manifest is missing.' }
$reader = New-Object IO.StreamReader($stream)
try { $manifest = $reader.ReadToEnd() } finally { $reader.Dispose() }
$resources = @()
foreach ($name in $assembly.GetManifestResourceNames()) {
    if ($name -eq 'DflPtWebUi.Launcher.Payload.Manifest') { continue }
    $resource = $assembly.GetManifestResourceStream($name)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = [BitConverter]::ToString($hasher.ComputeHash($resource)).Replace('-','').ToLowerInvariant()
        $resources += @{ resourceName=$name; size=$resource.Length; sha256=$digest }
    } finally { $resource.Dispose(); $hasher.Dispose() }
}
$version = [Diagnostics.FileVersionInfo]::GetVersionInfo($path)
@{
    assemblyName=$assembly.GetName().Name
    assemblyVersion=$assembly.GetName().Version.ToString()
    assemblyProduct=$attributes['AssemblyProductAttribute']
    assemblyFileVersion=$attributes['AssemblyFileVersionAttribute']
    fileProduct=$version.ProductName
    fileVersion=$version.FileVersion
    manifest=$manifest
    resources=$resources
} | ConvertTo-Json -Depth 6 -Compress
"""


def inspect_assembly(path: Path) -> dict:
    if os.name != "nt":
        raise PackageError("Assembly inspection requires Windows PowerShell / .NET Framework.")
    environment = os.environ.copy()
    environment["DFL_PT_PACKAGE_EXE"] = str(path)
    command = base64.b64encode(INSPECT_SCRIPT.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                             "-EncodedCommand", command], env=environment,
                            capture_output=True, timeout=60)
    if result.returncode:
        raise PackageError("Reflection-only launcher resource inspection failed.")
    try:
        return json.loads(result.stdout.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise PackageError("Launcher inspector returned invalid metadata.") from error


def safe_payload_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or "\0" in value
            or path.is_absolute() or str(path) != value
            or any(part in (".", "..", ".runtime", ".openai", "workspace", "workspaces")
                   for part in path.parts)):
        raise PackageError("Payload manifest contains an unsafe or private path.")
    return value


def parse_payload(manifest: str) -> tuple[str, list[dict]]:
    lines = [line for line in manifest.splitlines() if line]
    if not lines or not re.fullmatch(r"DFLSN_PAYLOAD_V1\t[0-9a-f]{24}", lines[0]):
        raise PackageError("Unsupported embedded launcher payload manifest.")
    entries = []
    paths, names = set(), set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 5 or fields[0] != "F":
            raise PackageError("Invalid launcher payload entry.")
        _, path, size, digest, name = fields
        safe_payload_path(path)
        if (not size.isdecimal() or int(size) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not re.fullmatch(re.escape(RESOURCE_PREFIX) + r"\d{4}", name)):
            raise PackageError("Invalid launcher payload hash, size or resource name.")
        if path.casefold() in paths or name in names:
            raise PackageError("Duplicate launcher payload path or resource.")
        paths.add(path.casefold())
        names.add(name)
        entries.append({"path": path, "size": int(size), "sha256": digest, "resourceName": name})
    material = "\n".join(f"{e['path']}\t{e['size']}\t{e['sha256']}" for e in entries)
    build_id = sha256(material.encode("utf-8"))[:24]
    if not entries or lines[0].split("\t")[1] != build_id:
        raise PackageError("Embedded payload build identity does not match its entries.")
    return build_id, entries


def ui_license_sources(root: Path) -> dict[str, Path]:
    ui = root / "launcher/ui"
    lock = json.loads((ui / "package-lock.json").read_text(encoding="utf-8-sig"))
    sources = {}
    for name, version in UI_RUNTIME_PACKAGES.items():
        package_root = ui / "node_modules" / name
        license_file = package_root / "LICENSE"
        metadata_file = package_root / "package.json"
        if not license_file.is_file() or not metadata_file.is_file():
            raise PackageError("A required bundled UI package or license is missing.")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
        locked = lock.get("packages", {}).get("node_modules/" + name, {})
        if (metadata.get("name") != name or metadata.get("version") != version
                or metadata.get("license") != "MIT" or locked.get("version") != version
                or locked.get("license") != "MIT"):
            raise PackageError("Bundled UI package does not match its pinned MIT dependency.")
        license_text = license_file.read_text(encoding="utf-8-sig")
        if (not 700 <= len(license_text) <= 65536
                or not re.search(r"copyright\s+\(c\)", license_text[:2048], re.IGNORECASE)
                or "Permission is hereby granted, free of charge" not in license_text
                or 'THE SOFTWARE IS PROVIDED "AS IS"' not in license_text):
            raise PackageError("Bundled UI license lacks its MIT text or copyright notice.")
        sources["licenses/UI/" + name + ".txt"] = license_file
    return sources


def payload_sources(root: Path) -> dict[str, Path]:
    sources = {name: root / source for name, source in FIXED_PAYLOAD.items()}
    sources.update(ui_license_sources(root))
    builder = (root / "launcher/build-host.ps1").read_text(encoding="utf-8-sig")
    match = re.search(r'\$WebView2Version\s*=\s*"([0-9.]+)"', builder)
    if not match:
        raise PackageError("Cannot determine the pinned WebView2 SDK source.")
    sdk = root / "launcher/vendor/webview2" / match[1]
    sources.update({
        "Microsoft.Web.WebView2.Core.dll": sdk / "lib/net462/Microsoft.Web.WebView2.Core.dll",
        "Microsoft.Web.WebView2.Wpf.dll": sdk / "lib/net462/Microsoft.Web.WebView2.Wpf.dll",
        "WebView2Loader.dll": sdk / "runtimes/win-x64/native/WebView2Loader.dll",
        "licenses/WebView2/LICENSE.txt": sdk / "LICENSE.txt",
        "licenses/WebView2/NOTICE.txt": sdk / "NOTICE.txt",
    })
    ui = root / "launcher/ui/dist/client"
    if not (ui / "index.html").is_file():
        raise PackageError("The tested launcher UI build is missing.")
    for path in ui.rglob("*"):
        if path.is_file():
            relative = path.relative_to(ui).as_posix()
            safe_payload_path("ui/" + relative)
            if not path.resolve().is_relative_to(ui.resolve()):
                raise PackageError("Launcher UI build contains an external linked file.")
            sources["ui/" + relative] = path
    return sources


def verify_payload(inspection: dict, sources: dict[str, Path], version: str) -> tuple[str, list[dict]]:
    expected_version = version.split("-", 1)[0] + ".0"
    for key in ("assemblyProduct", "fileProduct"):
        if inspection.get(key) != PRODUCT:
            raise PackageError("Launcher product identity does not match the PT project.")
    for key in ("assemblyVersion", "assemblyFileVersion", "fileVersion"):
        if inspection.get(key) != expected_version:
            raise PackageError("Launcher assembly/file version does not match the release.")
    if inspection.get("assemblyName") != "DFL-PT-WEBUI.Launcher":
        raise PackageError("Launcher assembly identity does not match the PT project.")
    build_id, entries = parse_payload(inspection.get("manifest", ""))
    resources = inspection.get("resources", [])
    if not isinstance(resources, list) or any(not isinstance(item, dict) for item in resources):
        raise PackageError("Invalid inspected resource inventory.")
    by_name = {item.get("resourceName"): item for item in resources}
    if len(by_name) != len(resources) or set(by_name) != {e["resourceName"] for e in entries}:
        raise PackageError("Executable resources do not exactly match its embedded manifest.")
    if {e["path"] for e in entries} != set(sources):
        raise PackageError("Embedded payload is incomplete or differs from the current build sources.")
    for entry in entries:
        inspected = by_name[entry["resourceName"]]
        source = sources[entry["path"]]
        if inspected.get("sha256") != entry["sha256"] or inspected.get("size") != entry["size"]:
            raise PackageError("An executable resource differs from its embedded checksum.")
        if (not source.is_file() or source.stat().st_size != entry["size"]
                or digest_file(source) != entry["sha256"]):
            raise PackageError("The executable embeds stale source or UI resources.")
    return build_id, entries


def build_source_paths(root: Path) -> dict[str, Path]:
    paths = [path for path in (root / "launcher/host").iterdir() if path.is_file()]
    paths.extend(root / relative for relative in (
        "launcher/build-host.ps1", "launcher/ui/package.json",
        "launcher/ui/package-lock.json", "launcher/ui/vite.config.mjs",
        "launcher/ui/index.html", "launcher/ui/scripts/prepare-sites-build.mjs"))
    npm_configuration = root / "launcher/ui/.npmrc"
    if npm_configuration.is_file():
        paths.append(npm_configuration)
    for name in UI_RUNTIME_PACKAGES:
        paths.extend(root / "launcher/ui/node_modules" / name / filename
                     for filename in ("package.json", "LICENSE"))
    for relative in ("launcher/ui/src", "launcher/ui/public"):
        directory = root / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): path for path in paths}


def verify_build_receipt(root: Path, executable: Path, build_id: str) -> list[dict]:
    receipt_path = executable.parent / "launcher-build.json"
    if not receipt_path.is_file() or receipt_path.stat().st_size > 2 * 1024 * 1024:
        raise PackageError("The compiler source receipt is missing or invalid; rebuild the launcher.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (receipt.get("schemaVersion") != 1 or receipt.get("product") != PRODUCT
            or receipt.get("payloadBuildId") != build_id):
        raise PackageError("Compiler receipt product or payload identity does not match.")
    if (receipt.get("uiBuildPerformed") is not True
            or receipt.get("sourceSnapshotBeforeUiBuild") is not True):
        raise PackageError("Standalone release requires a full UI build with sources captured before compilation.")
    if receipt.get("executable") != {"file": FILE_NAME, "sha256": digest_file(executable),
                                     "bytes": executable.stat().st_size}:
        raise PackageError("Compiler receipt belongs to a different executable.")
    recorded = receipt.get("sources")
    if not isinstance(recorded, list) or any(not isinstance(item, dict) for item in recorded):
        raise PackageError("Compiler receipt has no valid source inventory.")
    expected = build_source_paths(root)
    by_path = {}
    for item in recorded:
        name = safe_payload_path(item.get("path", ""))
        if name in by_path or name.casefold() in {path.casefold() for path in by_path}:
            raise PackageError("Duplicate compiler source receipt entry.")
        by_path[name] = item
    if set(by_path) != set(expected):
        raise PackageError("Compiler receipt omits current host or UI build inputs.")
    for name, path in expected.items():
        item = by_path[name]
        if (not path.is_file() or not path.resolve().is_relative_to(root.resolve())
                or type(item.get("bytes")) is not int or item["bytes"] != path.stat().st_size
                or item.get("sha256") != digest_file(path)):
            raise PackageError("Compiler receipt contains stale host or UI build inputs.")
    return [{key: item[key] for key in ("path", "bytes", "sha256")} for item in recorded]


def notices(root: Path) -> list[dict]:
    builder = (root / "launcher/build-host.ps1").read_text(encoding="utf-8-sig")
    version = re.search(r'\$WebView2Version\s*=\s*"([0-9.]+)"', builder)[1]
    records = [{"component": "Microsoft WebView2 SDK", "version": version,
             "source": "https://www.nuget.org/packages/Microsoft.Web.WebView2/" + version,
             "embeddedNotices": ["licenses/WebView2/LICENSE.txt", "licenses/WebView2/NOTICE.txt"]},
            {"component": PRODUCT, "license": "GPL-3.0",
             "source": REPOSITORY, "licenseUrl": REPOSITORY + "/blob/main/LICENSE"}]
    records.extend({"component": name, "version": runtime_version, "license": "MIT",
                    "source": "https://www.npmjs.com/package/" + name + "/v/" + runtime_version,
                    "embeddedNotices": ["licenses/UI/" + name + ".txt"]}
                   for name, runtime_version in UI_RUNTIME_PACKAGES.items())
    return records


def package(root: Path, source_revision: str, version: str, output: Path) -> dict:
    if not re.fullmatch(r"\d+\.\d+\.\d+-preview", version):
        raise PackageError("Use an explicit launcher preview version.")
    source = verified_source(root, source_revision)
    executable = root / "launcher/bin" / FILE_NAME
    verify_pe(executable)
    inspection = inspect_assembly(executable)
    build_id, entries = verify_payload(inspection, payload_sources(root), version)
    compiler_sources = verify_build_receipt(root, executable, build_id)
    output = output.resolve()
    release_root = (root / "release-output").resolve()
    if not output.is_relative_to(release_root) or output == release_root:
        raise PackageError("Use a new dedicated subdirectory inside release-output.")
    if output.exists() and any(output.iterdir()):
        raise PackageError("Existing release assets are never overwritten; use a fresh directory.")
    executable_hash = digest_file(executable)
    payload = [{key: e[key] for key in ("path", "size", "sha256")} for e in entries]
    provenance = {
        "schemaVersion": 1, "product": PRODUCT, "version": version,
        "releaseTag": "launcher-v" + version,
        "createdUtc": datetime.now(timezone.utc).isoformat(), "source": source,
        "launcher": {"file": FILE_NAME, "size": executable.stat().st_size,
                     "sha256": executable_hash, "machine": "AMD64", "format": "PE32+",
                     "assemblyVersion": inspection["assemblyVersion"],
                     "fileVersion": inspection["fileVersion"], "payloadBuildId": build_id},
        "payload": payload, "compilerSources": compiler_sources,
        "thirdPartyNotices": notices(root),
        "validation": {"reflectionOnly": True, "sourceTreeMatchesCheckout": True,
                       "embeddedPayloadMatchesSources": True,
                       "compilerInputsMatchSources": True,
                       "fullUiBuildPerformed": True,
                       "sourcesCapturedBeforeUiBuild": True,
                       "freshWindowsInstallation": "not claimed"},
    }
    # A concurrent source edit must not slip in after the initial identity check.
    if verified_source(root, source_revision) != source:
        raise PackageError("Public source identity changed while packaging.")
    output.mkdir(parents=True, exist_ok=True)
    target = output / FILE_NAME
    with executable.open("rb") as source_stream, target.open("xb") as destination:
        shutil.copyfileobj(source_stream, destination)
    if digest_file(target) != executable_hash:
        raise PackageError("Launcher bytes changed during packaging.")
    provenance_name = "DFL-PT-WEBUI.Launcher.provenance.json"
    provenance_data = (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with (output / provenance_name).open("xb") as stream:
        stream.write(provenance_data)
    sums = f"{executable_hash}  {FILE_NAME}\n{sha256(provenance_data)}  {provenance_name}\n"
    with (output / "SHA256SUMS.launcher").open("xb") as stream:
        stream.write(sums.encode("ascii"))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True,
                        help="Reviewed sanitized public commit whose tree matches this checkout.")
    parser.add_argument("--version", default="0.1.2-preview")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or ROOT / "release-output" / ("launcher-v" + arguments.version)
    provenance = package(ROOT, arguments.source_revision, arguments.version, output)
    print(json.dumps({"product": provenance["product"], "version": provenance["version"],
                      "sourceCommit": provenance["source"]["commit"],
                      "sha256": provenance["launcher"]["sha256"], "assets": 3}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (PackageError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print("Launcher packaging failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
