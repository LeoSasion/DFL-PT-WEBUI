"""Restore a selected batch into an independent complete faceset copy."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
from DFLIMG import DFLIMG
from core import pathex
from core.cv2ex import cv2_imread, cv2_imwrite
from core.interact import interact as io
from facelib.FaceEnhancement import AlignedFaceEnhancer, MODELS


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_folder(dirpath, cpu_only=False, force_gpu_idxs=None, model_id='mambairv2', offset=0, limit=500):
    directory = Path(dirpath).absolute()
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir() or directory.resolve() != directory:
        raise ValueError('A plain aligned directory is required')
    if model_id not in MODELS or isinstance(offset, bool) or isinstance(limit, bool) or not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError('Restoration requires a fixed model and a batch of 1–500 images')
    paths = [Path(item) for item in pathex.get_image_paths(directory)]
    if not paths or offset >= len(paths) or len({p.name.casefold() for p in paths}) != len(paths):
        raise ValueError('Nonempty unique faceset selection required')
    for source in paths:
        if source.is_symlink() or not source.is_file() or source.resolve().parent != directory:
            raise ValueError('Faceset contains linked/invalid images')
    selected = paths[offset:offset + limit]
    hashes = {source.name: sha256(source) for source in paths}
    sidecars = {}
    for source in paths:
        audit = source.with_suffix(source.suffix + '.landmarks.json')
        if audit.exists():
            if audit.is_symlink() or not audit.is_file() or audit.resolve().parent != directory:
                raise ValueError('Faceset contains a linked/invalid landmark audit')
            sidecars[source.name] = (audit, sha256(audit))
    metadata = {}
    for source in selected:
        dfl = DFLIMG.load(source)
        if dfl is None or not dfl.has_data():
            raise ValueError(f'{source.name}: valid aligned metadata required before restoration')
        metadata[source.name] = copy.deepcopy(dfl.get_dict())
    device = 'cpu' if cpu_only else 'cuda:' + str(int(str(force_gpu_idxs or '0').split(',')[0]))
    model = AlignedFaceEnhancer(model_id, device=device)
    batch_id = 'enhance-' + uuid.uuid4().hex
    root = directory.parent / (directory.name + '_enhanced')
    root.mkdir(exist_ok=True)
    if root.resolve() != root or root.is_symlink() or root.is_junction():
        raise ValueError('Enhancement history must be a plain directory')
    pending, output = root / ('.' + batch_id + '.pending'), root / batch_id
    pending.mkdir()
    receipt = {'schemaVersion': 1, 'id': batch_id, 'status': 'running', 'originalRetained': True,
               'independentCopies': True, 'subset': False, 'inputDirectory': str(directory),
               'selectedRange': {'start': offset + 1, 'end': offset + len(selected), 'total': len(paths), 'limit': 500},
               'model': model.provenance, 'needsReview': True, 'entries': []}
    def save():
        temporary = pending / 'enhancement.receipt.json.tmp'
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(pending / 'enhancement.receipt.json')
    try:
        chosen = {source.name for source in selected}
        for source in paths:
            destination = pending / source.name
            if source.name not in chosen:
                shutil.copyfile(source, destination)
                if source.name in sidecars:
                    shutil.copyfile(sidecars[source.name][0], pending / sidecars[source.name][0].name)
            else:
                image = cv2_imread(source)
                if image is None:
                    raise ValueError('Aligned decoding failed: ' + source.name)
                enhanced = model.enhance(image.astype(np.float32) / 255., preserve_size=True)
                cv2_imwrite(str(destination), np.rint(enhanced * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                dfl = DFLIMG.load(destination)
                if dfl is None:
                    raise RuntimeError('Enhanced image metadata container unsupported')
                data = copy.deepcopy(metadata[source.name])
                # Content-bound native audits are no longer predictions on these new pixels.
                data.pop('native_landmarks98', None)
                data['enhancement_provenance'] = {'inputSha256': hashes[source.name], 'model': model.provenance,
                    'geometryPreserved': True, 'annotationsRetainedForReview': True, 'native98AuditInvalidated': True}
                dfl.set_dict(data)
                dfl.save()
                checked = DFLIMG.load(destination)
                if not checked or not checked.has_data() or checked.get_shape()[:2] != image.shape[:2]:
                    raise RuntimeError('Enhanced aligned metadata/shape readback failed')
            receipt['entries'].append({'file': source.name, 'enhanced': source.name in chosen,
                'inputSha256': hashes[source.name], 'outputSha256': sha256(destination),
                'nativeAuditSidecar': {'presentInOriginal': source.name in sidecars,
                    'copiedUnchanged': source.name in sidecars and source.name not in chosen,
                    'invalidatedByNewPixels': source.name in sidecars and source.name in chosen}})
            save()
            io.log_info(f"[Enhancement] {len(receipt['entries'])}/{len(paths)} 独立副本")
        if (any(sha256(source) != hashes[source.name] for source in paths)
            or any(sha256(audit) != digest for audit, digest in sidecars.values())
            or {source.name for source in paths if source.with_suffix(source.suffix + '.landmarks.json').exists()} != set(sidecars)
            or {Path(p).name for p in pathex.get_image_paths(directory)} != set(hashes)):
            raise RuntimeError('Source faceset changed before enhancement publication')
        receipt.update(status='completed', facesetPath=str(output), selectedCount=len(selected), copiedCount=len(paths))
        save()
        pending.rename(output)
        io.log_info(f'已生成完整独立人脸集：{output}；原件保留，增强结果须复核。')
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise
