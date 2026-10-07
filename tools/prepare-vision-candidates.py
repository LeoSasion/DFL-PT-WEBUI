"""Explicit acquisition of hash-pinned candidate weights; no inference or installation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from vision_assets import DEFAULT_ASSETS_ROOT, atomic_json, file_sha256


def acquire(model_ids, *, assets_root=DEFAULT_ASSETS_ROOT, allow_download=False, import_root=None):
    root = Path(assets_root).resolve()
    catalog = json.loads((ROOT / "tools" / "vision-model-candidates.json").read_text(encoding="utf-8"))
    declared = {item["id"]: item for item in catalog["detectorAssets"]}
    if not model_ids or any(model_id not in declared for model_id in model_ids):
        raise ValueError("Only declared detector candidate IDs may be acquired")
    registry_path = root / "detectors" / "assets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"schemaVersion": 1, "models": []}
    models = {model["id"]: model for model in registry["models"]}
    for model_id in model_ids:
        entry = dict(declared[model_id])
        target = root / "detectors" / entry["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            pending = target.with_suffix(target.suffix + ".partial")
            source = Path(import_root) / entry["filename"] if import_root else None
            try:
                if source and source.is_file():
                    shutil.copyfile(source, pending)
                elif allow_download:
                    request = urllib.request.Request(entry["source"], headers={"User-Agent": "DFL-PT-WEBUI-model-acquisition/1"})
                    with urllib.request.urlopen(request, timeout=60) as incoming, pending.open("wb") as output:
                        shutil.copyfileobj(incoming, output, length=1024 * 1024)
                else:
                    raise ValueError(f"Missing {model_id}; acquire explicitly with --allow-download or --import-root")
                if pending.stat().st_size != entry["sizeBytes"] or file_sha256(pending) != entry["sha256"]:
                    raise ValueError(f"Candidate byte count or SHA-256 mismatch: {model_id}")
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
        if target.stat().st_size != entry["sizeBytes"] or file_sha256(target) != entry["sha256"]:
            raise ValueError(f"Existing candidate differs; it has not been overwritten: {model_id}")
        entry.update(path=target.relative_to(root).as_posix(), acquired=True,
                     admission="evaluation-only", default=False)
        models[model_id] = entry
        registry["models"] = sorted(models.values(), key=lambda value: value["id"])
        atomic_json(registry_path, registry)
        print(json.dumps({"id": model_id, "sha256": entry["sha256"], "status": "verified-local-evaluation-asset"}), flush=True)
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Declared candidate IDs")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--import-root", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    acquire(args.models, assets_root=args.assets_root, allow_download=args.allow_download,
            import_root=args.import_root)
