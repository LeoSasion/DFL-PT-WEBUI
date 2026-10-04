"""Launcher distribution boundary tests; no launcher execution or installation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "package_launcher", Path(__file__).resolve().parents[1] / "package-launcher.py")
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)


def fixture_pe(machine=0x8664, magic=0x20B):
    data = bytearray(2048)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 132, machine)
    struct.pack_into("<H", data, 152, magic)
    return bytes(data)


def fixture_inspection(sources):
    rows, resources = [], []
    for index, (name, path) in enumerate(sources.items()):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        resource_name = PACKAGER.RESOURCE_PREFIX + f"{index:04d}"
        rows.append(f"F\t{name}\t{len(content)}\t{digest}\t{resource_name}")
        resources.append({"resourceName": resource_name, "size": len(content), "sha256": digest})
    material = "\n".join("\t".join(row.split("\t")[1:4]) for row in rows)
    build_id = hashlib.sha256(material.encode()).hexdigest()[:24]
    return {"assemblyProduct": PACKAGER.PRODUCT, "fileProduct": PACKAGER.PRODUCT,
            "assemblyName": "DFL-PT-WEBUI.Launcher", "assemblyVersion": "0.1.2.0",
            "assemblyFileVersion": "0.1.2.0", "fileVersion": "0.1.2.0",
            "manifest": "DFLSN_PAYLOAD_V1\t" + build_id + "\n" + "\n".join(rows),
            "resources": resources}


def fixture_ui_licenses(root):
    ui = root / "launcher/ui"
    lock = {"packages": {"": {"name": "fixture-launcher", "version": "0.1.0"}}}
    text = '''MIT License

Copyright (c) Fixture authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''
    for name, version in PACKAGER.UI_RUNTIME_PACKAGES.items():
        package = ui / "node_modules" / name
        package.mkdir(parents=True, exist_ok=True)
        metadata = {"name": name, "version": version, "license": "MIT"}
        (package / "package.json").write_text(json.dumps(metadata), encoding="utf-8")
        (package / "LICENSE").write_text(text, encoding="utf-8")
        lock["packages"]["node_modules/" + name] = metadata
    (ui / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")


class PayloadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = {}
        for index, name in enumerate([*PACKAGER.FIXED_PAYLOAD, "ui/index.html",
                                      "licenses/WebView2/LICENSE.txt", "licenses/WebView2/NOTICE.txt",
                                      *("licenses/UI/" + name + ".txt" for name in PACKAGER.UI_RUNTIME_PACKAGES)]):
            file = self.root / f"input-{index}"
            file.write_bytes(("verified source: " + name).encode())
            self.sources[name] = file
        self.inspection = fixture_inspection(self.sources)

    def test_valid_payload_requires_current_resource_bytes(self):
        build, entries = PACKAGER.verify_payload(self.inspection, self.sources, "0.1.2-preview")
        self.assertEqual(len(build), 24)
        self.assertEqual(len(entries), len(self.sources))
        self.sources["bootstrap/install-project.ps1"].write_bytes(b"changed after build")
        with self.assertRaisesRegex(PACKAGER.PackageError, "stale"):
            PACKAGER.verify_payload(self.inspection, self.sources, "0.1.2-preview")

    def test_rejects_missing_installer_and_license_payloads(self):
        for omitted in ["bootstrap/install-project.ps1", "bootstrap/install-source.ps1",
                        "licenses/WebView2/LICENSE.txt",
                        *("licenses/UI/" + name + ".txt" for name in PACKAGER.UI_RUNTIME_PACKAGES)]:
            with self.subTest(omitted=omitted):
                missing = dict(self.sources)
                del missing[omitted]
                with self.assertRaisesRegex(PACKAGER.PackageError, "incomplete"):
                    PACKAGER.verify_payload(fixture_inspection(missing), self.sources, "0.1.2-preview")
        fixture_ui_licenses(self.root)
        checked = PACKAGER.ui_license_sources(self.root)
        self.assertEqual(len(checked), len(PACKAGER.UI_RUNTIME_PACKAGES))
        license_file = self.root / "launcher/ui/node_modules/react/LICENSE"
        license_bytes = license_file.read_bytes()
        license_file.unlink()
        with self.assertRaisesRegex(PACKAGER.PackageError, "missing"):
            PACKAGER.ui_license_sources(self.root)
        license_file.write_bytes(license_bytes)
        package_json = license_file.parent / "package.json"
        metadata = json.loads(package_json.read_text(encoding="utf-8"))
        package_json.write_text(json.dumps({**metadata, "version": "0.0.0"}), encoding="utf-8")
        with self.assertRaisesRegex(PACKAGER.PackageError, "pinned MIT"):
            PACKAGER.ui_license_sources(self.root)
        package_json.write_text(json.dumps(metadata), encoding="utf-8")
        license_file.write_bytes(license_bytes.replace(b"Copyright", b"Unrelated"))
        with self.assertRaisesRegex(PACKAGER.PackageError, "copyright"):
            PACKAGER.ui_license_sources(self.root)

    def test_rejects_tampered_embedded_bytes_even_with_valid_manifest(self):
        self.inspection["resources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PACKAGER.PackageError, "checksum"):
            PACKAGER.verify_payload(self.inspection, self.sources, "0.1.2-preview")

    def test_rejects_unlisted_or_duplicate_executable_resources(self):
        self.inspection["resources"].append(dict(self.inspection["resources"][0]))
        with self.assertRaisesRegex(PACKAGER.PackageError, "exactly match"):
            PACKAGER.verify_payload(self.inspection, self.sources, "0.1.2-preview")

    def test_rejects_other_product_and_stale_version(self):
        for key, value in [("assemblyProduct", "DeepFaceLab-WEBUI Launcher"),
                           ("assemblyFileVersion", "0.1.1.0")]:
            with self.subTest(key=key):
                wrong = {**self.inspection, key: value}
                with self.assertRaises(PACKAGER.PackageError):
                    PACKAGER.verify_payload(wrong, self.sources, "0.1.2-preview")

    def test_rejects_private_and_traversal_payload_paths(self):
        for path in ["../settings.json", "C:/user/key.json", "ui/.runtime/settings.json",
                     "ui/.openai/hosting.json", "ui/workspace/face.jpg", "ui\\index.html"]:
            with self.subTest(path=path), self.assertRaises(PACKAGER.PackageError):
                PACKAGER.safe_payload_path(path)

    def test_rejects_duplicate_manifest_paths_and_forged_build_id(self):
        with self.assertRaisesRegex(PACKAGER.PackageError, "Duplicate"):
            PACKAGER.parse_payload(self.inspection["manifest"] + "\n" +
                                   self.inspection["manifest"].splitlines()[1])
        forged = self.inspection["manifest"].replace(
            self.inspection["manifest"].splitlines()[0], "DFLSN_PAYLOAD_V1\t" + "0" * 24)
        with self.assertRaisesRegex(PACKAGER.PackageError, "build identity"):
            PACKAGER.parse_payload(forged)

    def test_pe_checks_machine_and_format_without_executing_it(self):
        executable = self.root / "launcher.exe"
        executable.write_bytes(fixture_pe())
        PACKAGER.verify_pe(executable)
        for data in [fixture_pe(machine=0x14C, magic=0x10B), b"MZ" + bytes(2046), b"not PE"]:
            executable.write_bytes(data)
            with self.assertRaises(PACKAGER.PackageError):
                PACKAGER.verify_pe(executable)


class SourceAndPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.command("init", "--quiet")
        (self.root / "README.md").write_text("PT fixture\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("release-output/\nlauncher/bin/\n", encoding="utf-8")
        self.command("add", ".")
        self.command("commit", "--quiet", "-m", "sanitized public snapshot")
        self.public_commit = self.command("rev-parse", "HEAD")
        self.command("update-ref", "refs/remotes/public/main", self.public_commit)
        self.command("remote", "add", "public", PACKAGER.REPOSITORY + ".git")

    def command(self, *args):
        result = subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                                 "-c", "user.email=fixture@localhost", *args],
                                check=True, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip()

    def test_accepts_equal_tree_sanitized_snapshot_from_private_checkout(self):
        self.command("checkout", "--quiet", "--orphan", "private")
        self.command("commit", "--quiet", "-m", "separate private history")
        verified = PACKAGER.verified_source(self.root, self.public_commit)
        self.assertEqual(verified["commit"], self.public_commit)
        self.assertEqual(verified["tree"], self.command("rev-parse", "HEAD^{tree}"))
        with self.assertRaises(PACKAGER.PackageError):
            PACKAGER.verified_source(self.root, "HEAD")

    def test_rejects_public_commit_merging_private_ancestry(self):
        self.command("checkout", "--quiet", "--orphan", "private")
        self.command("commit", "--quiet", "-m", "private ancestry")
        self.command("merge", "--quiet", "--allow-unrelated-histories", "--no-edit", self.public_commit)
        with self.assertRaisesRegex(PACKAGER.PackageError, "unrelated history"):
            PACKAGER.verified_source(self.root, "HEAD")

    def test_rejects_wrong_remote_dirty_tree_and_mismatched_revision(self):
        self.command("remote", "set-url", "public", "https://github.com/LeoSasion/DeepFaceLab-WEBUI.git")
        with self.assertRaisesRegex(PACKAGER.PackageError, "official PT"):
            PACKAGER.verified_source(self.root, self.public_commit)
        self.command("remote", "set-url", "public", PACKAGER.REPOSITORY + ".git")
        (self.root / "README.md").write_text("new reviewed source\n", encoding="utf-8")
        with self.assertRaisesRegex(PACKAGER.PackageError, "Commit"):
            PACKAGER.verified_source(self.root, self.public_commit)
        self.command("add", ".")
        self.command("commit", "--quiet", "-m", "source change")
        with self.assertRaisesRegex(PACKAGER.PackageError, "trees differ"):
            PACKAGER.verified_source(self.root, self.public_commit)

    def test_packages_exact_bytes_without_local_paths_and_preserves_existing_release(self):
        executable = self.root / "launcher/bin" / PACKAGER.FILE_NAME
        executable.parent.mkdir(parents=True)
        executable.write_bytes(fixture_pe())
        source = self.root / "README.md"
        payload = {"ui/index.html": source}
        inspection = fixture_inspection(payload)
        output = self.root / "release-output/launcher-v0.1.2-preview"
        with patch.object(PACKAGER, "inspect_assembly", return_value=inspection), \
                patch.object(PACKAGER, "payload_sources", return_value=payload), \
                patch.object(PACKAGER, "verify_build_receipt", return_value=[]), \
                patch.object(PACKAGER, "notices", return_value=[]):
            provenance = PACKAGER.package(self.root, self.public_commit, "0.1.2-preview", output)
            self.assertEqual((output / PACKAGER.FILE_NAME).read_bytes(), executable.read_bytes())
            self.assertEqual({p.name for p in output.iterdir()},
                             {PACKAGER.FILE_NAME, "SHA256SUMS.launcher", "DFL-PT-WEBUI.Launcher.provenance.json"})
            serialized = json.dumps(provenance)
            self.assertNotIn(str(self.root), serialized)
            self.assertNotIn("fixture@localhost", serialized)
            for line in (output / "SHA256SUMS.launcher").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, PACKAGER.digest_file(output / name))
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(PACKAGER.PackageError, "never overwritten"):
                PACKAGER.package(self.root, self.public_commit, "0.1.2-preview", output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


class CompilerReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in ["launcher/host/Program.cs", "launcher/host/AssemblyInfo.cs",
                         "launcher/host/app.manifest", "launcher/build-host.ps1",
                         "launcher/ui/package.json", "launcher/ui/package-lock.json",
                         "launcher/ui/vite.config.mjs", "launcher/ui/index.html",
                         "launcher/ui/scripts/prepare-sites-build.mjs", "launcher/ui/src/App.jsx",
                         "launcher/ui/public/assets/brand.png"]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("compiler input " + relative).encode())
        fixture_ui_licenses(self.root)
        self.executable = self.root / "launcher/bin" / PACKAGER.FILE_NAME
        self.executable.parent.mkdir(parents=True)
        self.executable.write_bytes(fixture_pe())
        self.build_id = "a" * 24
        self.receipt = {
            "schemaVersion": 1, "product": PACKAGER.PRODUCT, "payloadBuildId": self.build_id,
            "uiBuildPerformed": True, "sourceSnapshotBeforeUiBuild": True,
            "executable": {"file": PACKAGER.FILE_NAME, "bytes": self.executable.stat().st_size,
                           "sha256": PACKAGER.digest_file(self.executable)},
            "sources": [{"path": relative, "bytes": path.stat().st_size,
                         "sha256": PACKAGER.digest_file(path)}
                        for relative, path in PACKAGER.build_source_paths(self.root).items()],
        }
        self.write_receipt()

    def write_receipt(self):
        (self.executable.parent / "launcher-build.json").write_text(
            json.dumps(self.receipt), encoding="utf-8")

    def test_current_compiler_receipt_covers_host_and_ui_inputs(self):
        checked = PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)
        self.assertEqual({item["path"] for item in checked}, set(PACKAGER.build_source_paths(self.root)))

    def test_omitted_host_file_does_not_certify_executable(self):
        self.receipt["sources"] = [item for item in self.receipt["sources"]
                                   if item["path"] != "launcher/host/Program.cs"]
        self.write_receipt()
        with self.assertRaisesRegex(PACKAGER.PackageError, "omits"):
            PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)

    def test_host_change_after_compilation_requires_rebuild(self):
        path = self.root / "launcher/host/Program.cs"
        path.write_bytes(path.read_bytes().replace(b"compiler", b"modified"))
        with self.assertRaisesRegex(PACKAGER.PackageError, "stale"):
            PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)

    def test_added_npm_configuration_requires_new_compiler_receipt(self):
        (self.root / "launcher/ui/.npmrc").write_text("registry=https://registry.npmjs.org\n")
        with self.assertRaisesRegex(PACKAGER.PackageError, "omits"):
            PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)

    def test_skip_ui_build_or_missing_prebuild_snapshot_cannot_certify_release(self):
        for key in ("uiBuildPerformed", "sourceSnapshotBeforeUiBuild"):
            for value in (False, None, "true", 1):
                with self.subTest(key=key, value=value):
                    previous = self.receipt[key]
                    self.receipt[key] = value
                    self.write_receipt()
                    with self.assertRaisesRegex(PACKAGER.PackageError, "full UI build"):
                        PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)
                    self.receipt[key] = previous

    def test_rejects_other_executable_and_payload_identity(self):
        self.executable.write_bytes(fixture_pe() + b"tampered")
        with self.assertRaisesRegex(PACKAGER.PackageError, "different executable"):
            PACKAGER.verify_build_receipt(self.root, self.executable, self.build_id)
        with self.assertRaisesRegex(PACKAGER.PackageError, "identity"):
            PACKAGER.verify_build_receipt(self.root, self.executable, "b" * 24)


if __name__ == "__main__":
    unittest.main()
