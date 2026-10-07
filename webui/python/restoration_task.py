"""Source-frame restoration into an atomic new copy; never mutate aligned media."""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np
from PIL import Image

from vision_restoration import RestorationModel

TASK_ID = re.compile(r"rst-[a-f0-9]{32}\Z")
IMAGE_NAME = re.compile(r'[^<>:"/\\|?*\x00-\x1f]{1,220}\.(?:png|jpe?g)\Z', re.I)
MAX_IMAGES = 500
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SOURCE_MODELS = ("swinir-psnr", "realesrgan-x4plus")


def plain_path(root, target, directory=False):
    root, target = Path(root).absolute(), Path(target).absolute()
    if not target.is_relative_to(root):
        raise ValueError("Restoration path escapes its workspace")
    current = root
    for part in [None, *target.relative_to(root).parts]:
        if part is not None:
            current /= part
        if current.is_symlink() or current.is_junction():
            raise ValueError("Restoration paths must not contain links/junctions")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Restoration path escapes its canonical workspace")
    if directory and not target.is_dir():
        raise ValueError("Required restoration directory missing")
    return target


def read_source(workspace, side, item):
    if side not in ("src", "dst") or not IMAGE_NAME.fullmatch(item.get("name", "")):
        raise ValueError("Only selected top-level source frames are accepted")
    target = plain_path(workspace, Path(workspace) / f"data_{side}" / item["name"])
    info = target.lstat()
    if not target.is_file() or info.st_size > MAX_IMAGE_BYTES or info.st_size < 1:
        raise ValueError("Source must be a plain image no larger than 50 MiB")
    with target.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(MAX_IMAGE_BYTES + 1)
    after = target.lstat()
    if ((opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)
            or len(data) > MAX_IMAGE_BYTES):
        raise ValueError("Source changed during verified reading")
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("Source bytes changed since task selection")
    return data


def assert_source_container(data):
    """Reject executable DFL metadata without deserializing pickle payloads."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        while offset + 12 <= len(data):
            length = struct.unpack(">I", data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            end = offset + length + 12
            if end > len(data):
                raise ValueError("Malformed PNG container")
            if kind in (b"fcWp", b"dfLa", b"dflD"):
                raise ValueError("Aligned metadata is unsupported; restore source frames then extract again")
            if kind == b"IHDR" and (length != 13 or data[offset+16] != 8):
                raise ValueError("Restoration accepts RGB8 images, not high-bit-depth PNG")
            offset = end
            if kind == b"IEND":
                return
        raise ValueError("Incomplete PNG container")
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset < len(data):
            if data[offset] != 255:
                raise ValueError("Malformed JPEG marker")
            while offset < len(data) and data[offset] == 255:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker == 239:
                raise ValueError("Aligned APP15 metadata unsupported; restore source frames then extract again")
            if marker in (218, 217):
                return
            if marker == 1 or 208 <= marker <= 215:
                continue
            if offset + 2 > len(data):
                break
            length = struct.unpack(">H", data[offset:offset+2])[0]
            if length < 2 or offset + length > len(data):
                break
            offset += length
        raise ValueError("Incomplete JPEG container")
    raise ValueError("Only JPEG/PNG source frames are supported")


def decode_source(data):
    assert_source_container(data)
    with Image.open(BytesIO(data)) as image:
        if (image.mode not in ("RGB", "L") or image.width * image.height > MAX_IMAGE_PIXELS
                or min(image.size) < 32 or getattr(image, "n_frames", 1) != 1):
            raise ValueError("Static RGB8 source frames between 32 pixels and 16 MP required")
        if image.info.get("icc_profile") or image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize color profile/orientation before restoration")
        return np.asarray(image.convert("RGB")).copy()


def run_task(request, workspace, runtime, assets, model_factory=RestorationModel, progress=None):
    workspace, runtime = Path(workspace).absolute(), Path(runtime).absolute()
    plain_path(workspace, runtime, directory=True)
    if runtime != workspace / ".webui":
        raise ValueError("Restoration runtime must be the active workspace .webui")
    task_id, side = request.get("taskId", ""), request.get("side")
    items = request.get("inputs")
    model_id = request.get("modelId", "swinir-psnr")
    if not TASK_ID.fullmatch(task_id) or model_id not in SOURCE_MODELS:
        raise ValueError("Source-frame restoration supports one SwinIR/Real-ESRGAN model; GFPGAN requires separately qualified FFHQ alignment")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_IMAGES:
        raise ValueError("Select 1..500 source frames per restoration task")
    names = [item.get("name", "") for item in items]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Duplicate selected source frames")
    output_names = [Path(name).stem + ".png" for name in names]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("Selected source names collide after PNG conversion")
    for item in items:
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("Selected source SHA-256 required")
        decode_source(read_source(workspace, side, item))
    root = plain_path(workspace, runtime / "restoration", directory=True)
    staging_root = plain_path(workspace, root / "staging", directory=True)
    outputs_root = plain_path(workspace, root / "outputs", directory=True)
    staging, final = staging_root / task_id, outputs_root / task_id
    if staging.exists() or final.exists():
        raise ValueError("Restoration task outputs are immutable; use a new task")
    staging.mkdir()
    images = staging / "images"
    images.mkdir()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = model_factory(model_id, assets, device=request.get("device", "cuda"))
    records = []
    for index, (item, output_name) in enumerate(zip(items, output_names)):
        source = decode_source(read_source(workspace, side, item))
        height, width = source.shape[:2]
        restored = model.restore(source, output_size=(width, height))
        if restored.dtype != np.uint8 or restored.shape != source.shape:
            raise ValueError("Restoration must return same-size RGB uint8")
        target = images / output_name
        Image.fromarray(restored).save(target, format="PNG")
        output_bytes = target.read_bytes()
        decoded = decode_source(output_bytes)
        if decoded.shape != source.shape or not np.array_equal(decoded, restored):
            raise ValueError("Restoration PNG failed pixel/size roundtrip")
        records.append({"sourceName": item["name"], "sourceSha256": item["sha256"],
                        "inputSha256": item["sha256"], "outputSha256": hashlib.sha256(output_bytes).hexdigest(),
                        "name": output_name, "sha256": hashlib.sha256(output_bytes).hexdigest(),
                        "width": width, "height": height, "mode": "RGB", "metadataCopied": False})
        if progress:
            progress({"completed": index + 1, "total": len(items), "name": item["name"]})
    # Recheck all selected originals before publishing the batch in one rename.
    for item in items:
        read_source(workspace, side, item)
    manifest = {"schemaVersion": 1, "taskId": task_id, "side": side, "stage": "source-frames",
                "modelId": model_id, "selectedCount": len(items), "limit": MAX_IMAGES,
                "modelProvenance": {**model.provenance,
                                    "cleanSrPolicy": "native-x4-then-area-resize-to-source-width-height",
                                    "outputSizePolicy": "preserve-source-width-height"},
                "files": records, "outputs": records,
                "originalsPreserved": True, "atomicBatch": True, "metadataCopied": False,
                "outputPolicy": "same-size RGB8 PNG; native x4 then area resize; no aligned coordinates copied",
                "nextStep": "Extract faces again from these source copies; never import directly into aligned"}
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plain_path(workspace, staging, directory=True)
    plain_path(workspace, outputs_root, directory=True)
    if final.exists():
        raise ValueError("Restoration output exists; refuse overwrite")
    staging.rename(final)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    args = parser.parse_args()
    request_file = plain_path(args.workspace, args.request)
    request = json.loads(request_file.read_text(encoding="utf-8"))
    def progress(value):
        print("RESTORATION_PROGRESS " + json.dumps(value, ensure_ascii=False), file=sys.stderr, flush=True)
    manifest = run_task(request, args.workspace, args.runtime, args.assets, progress=progress)
    print(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
