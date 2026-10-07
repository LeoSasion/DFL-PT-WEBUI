"""Local model identity and content-bound stage records; inference never downloads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = PROJECT_ROOT / "workspace" / ".vision-models"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def prepare_evaluation_output(output):
    """Keep locked comparison evidence separate from source images and assets."""
    output = Path(output).resolve()
    roots = (PROJECT_ROOT / "workspace/.vision-evaluation", PROJECT_ROOT / "workspace/.vision-models")
    if not any(output != root.resolve() and output.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Comparison outputs must be in an ignored project evaluation directory")
    if output.exists():
        raise ValueError("Use a fresh comparison output directory; previous evidence is immutable")
    output.mkdir(parents=True, exist_ok=False)
    return output


def atomic_json(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def resolve_asset_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Model paths must be relative to the local assets directory")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Model path escapes the local assets directory")
    return target


def verified_asset(model_id, *, assets_root=None):
    from vision_resource_paths import resource_group
    installed = resource_group(model_id, DEFAULT_ASSETS_ROOT) if assets_root is None else Path(assets_root)
    root = installed.resolve()
    registry_file = root / "assets.json" if assets_root is None and installed != DEFAULT_ASSETS_ROOT else root / "detectors" / "assets.json"
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if registry.get("schemaVersion") != 1:
        raise ValueError("Unsupported local detector asset registry")
    entries = [entry for entry in registry["models"] if entry.get("id") == model_id]
    if len(entries) != 1:
        raise ValueError(f"Model {model_id} is not acquired; run tools/prepare-vision-candidates.py")
    entry = entries[0]
    path = resolve_asset_path(root, entry["path"])
    if not path.is_file() or path.stat().st_size != entry["sizeBytes"]:
        raise ValueError(f"Missing or changed model bytes: {model_id}")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Model checksum mismatch: {model_id}")
    return path, entry


class StageCache:
    """Cache only successful, complete results with an exact dependency contract."""
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _identity(self, stage, dependencies):
        if not isinstance(stage, str) or not stage.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid cache stage")
        contract = {"schemaVersion": 1, "stage": stage, "dependencies": dependencies}
        key = canonical_digest(contract)
        return contract, key, self.root / stage / (key + ".json")

    def read(self, stage, dependencies):
        contract, key, path = self._identity(stage, dependencies)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("status") != "complete" or record.get("contract") != contract
                    or record.get("key") != key
                    or canonical_digest(record["result"]) != record.get("resultDigest")):
                return None
            return record["result"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, stage, dependencies, result):
        contract, key, path = self._identity(stage, dependencies)
        atomic_json(path, {"schemaVersion": 1, "status": "complete", "contract": contract,
                           "key": key, "resultDigest": canonical_digest(result), "result": result})
        return key
