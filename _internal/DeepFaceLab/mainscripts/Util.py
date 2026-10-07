import pickle
import hashlib
import json
import os
import uuid
from pathlib import Path

import cv2

from DFLIMG import *
from facelib import LandmarksProcessor, FaceType
from core.interact import interact as io
from core import pathex
from core.cv2ex import *
from core.safe_pickle import load_file as load_data_pickle_file, loads as load_data_pickle, validate_metadata_backup
from core.faceset_transaction import execute_plan, sha256


def save_faceset_metadata_folder(input_path):
    input_path = Path(input_path).absolute()
    metadata_filepath = input_path / 'meta.dat'
    manifest_filepath = input_path / 'meta.dat.manifest.json'
    io.log_info(f'正在保存元数据到 {metadata_filepath}\n')
    data, entries = {}, []
    for filename in io.progress_bar_generator(pathex.get_image_paths(input_path), '处理中'):
        filepath = Path(filename)
        image = DFLIMG.load(filepath)
        if image is None or not image.has_data():
            io.log_info(f'{filepath} 不是 DFL 图像文件')
            continue
        data[filepath.name] = (image.get_shape(), image.get_dict())
        entries.append({'filename': filepath.name, 'sha256': sha256(filepath), 'shape': list(image.get_shape())})
    if not data:
        raise ValueError('没有可保存元数据的 DFL 图片；保留现有备份')
    payload = pickle.dumps(data, protocol=4)
    validate_metadata_backup(load_data_pickle(payload, profile='session'))
    manifest = json.dumps({'schema': 1, 'metadata_sha256': hashlib.sha256(payload).hexdigest(),
                           'entries': entries}, ensure_ascii=False, indent=2).encode('utf-8')
    changes = []
    for path, content in ((metadata_filepath, payload), (manifest_filepath, manifest)):
        if path.exists():
            changes.append({'source': path.name, 'target': path.name, 'payload': content})
    if changes:
        receipt = execute_plan(input_path, changes, operation='metadata-save')
        io.log_info(f'前一次备份已归档：{receipt["receipt_path"]}')
    for path, content in ((metadata_filepath, payload), (manifest_filepath, manifest)):
        if any(change['source'] == path.name for change in changes): continue
        pending = input_path / ('.metadata-' + uuid.uuid4().hex + '.tmp')
        try:
            pending.write_bytes(content)
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    io.log_info('保持目录内文件名不变；恢复会生成完整原件备份和回执，meta.dat 会保留。')
    return {'metadata_path': str(metadata_filepath), 'count': len(data), 'manifest_path': str(manifest_filepath)}


def restore_faceset_metadata_folder(input_path, *, dry_run=False):
    input_path = Path(input_path).absolute()
    metadata_filepath = input_path / 'meta.dat'
    io.log_info(f'正在从 {metadata_filepath} 恢复元数据。\n')
    if not metadata_filepath.is_file():
        raise FileNotFoundError(metadata_filepath)
    try:
        data = validate_metadata_backup(load_data_pickle_file(metadata_filepath, profile='session'))
    except (OSError, ValueError) as error:
        raise ValueError(f'无法读取 faceset 元数据：{metadata_filepath}: {error}') from error
    metadata_sha = sha256(metadata_filepath)
    manifest_path = input_path / 'meta.dat.manifest.json'
    if manifest_path.exists():
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('元数据来源记录无效；未修改任何原件')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema') != 1 or manifest.get('metadata_sha256') != metadata_sha:
            raise ValueError('meta.dat 与来源 SHA 记录不一致；未修改任何原件')
    paths = {Path(name).name: Path(name) for name in pathex.get_image_paths(
        input_path, image_extensions=['.jpg', '.jpeg'], return_Path_class=True)}
    missing = sorted(set(data) - set(paths))
    if missing:
        raise ValueError('备份中的图片缺失；未修改任何原件：' + ', '.join(missing[:20]))
    changes, resized = [], []
    for name, (shape, metadata) in data.items():
        filepath = paths[name]
        image = cv2_imread(filepath)
        if image is None:
            raise ValueError(f'无法读取图片：{filepath}')
        if tuple(image.shape) != tuple(shape):
            image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not ok: raise ValueError(f'无法生成恢复图片：{filepath}')
            dflimg = DFLJPG.load(filepath, loader_func=lambda _: encoded.tobytes())
            resized.append(name)
        else:
            dflimg = DFLJPG.load(filepath)
        if dflimg is None:
            raise ValueError(f'恢复需要可读取的 JPEG：{filepath}')
        dflimg.set_dict(metadata)
        payload = dflimg.dump()
        verified = DFLJPG.load(filepath, loader_func=lambda _: payload)
        if verified is None or tuple(verified.get_shape()) != tuple(shape):
            raise ValueError(f'恢复 metadata 验证失败：{filepath}')
        changes.append({'source': name, 'target': name, 'payload': payload})
        sidecar = filepath.with_suffix(filepath.suffix + '.landmarks.json')
        if sidecar.exists() and payload != filepath.read_bytes():
            # Original sidecars describe another pixel SHA: archive them,
            # without turning old inference into a new prediction claim.
            changes.append({'source': sidecar.name, 'target': None})
    receipt = execute_plan(input_path, changes, operation='metadata-restore', dry_run=dry_run,
                           details={'metadata_sha256': metadata_sha, 'restored_count': len(data),
                                    'resized': resized, 'old_audit_sidecars_archived': [c['source'] for c in changes if c['target'] is None]})
    receipt['metadata_path'] = str(metadata_filepath)
    receipt['resized'] = resized
    io.log_info(f'恢复预览：{len(data)} 张，需缩放 {len(resized)} 张。' if dry_run else
                f'元数据整批恢复成功；备份文件保留。原件和恢复回执：{receipt["receipt_path"]}')
    return receipt

def add_landmarks_debug_images(input_path):
    io.log_info ("正在添加 landmarks 调试图...")

    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path), "处理中"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            continue

        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            face_type = FaceType.fromString ( dflimg.get_face_type() )

            if face_type == FaceType.MARK_ONLY:
                rect = dflimg.get_source_rect()
                LandmarksProcessor.draw_rect_landmarks(img, rect, face_landmarks, FaceType.FULL )
            else:
                LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True )



            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path, *, dry_run=False):
    input_path = Path(input_path).absolute()
    io.log_info ("正在恢复 aligned 原始文件名...")

    files = []
    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path), "处理中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False] ]

    files_len = len(files)
    for i in io.progress_bar_generator(range(files_len), "排序中"):
        fp, _, sf, converted = files[i]

        if converted:
            continue

        if not isinstance(sf, str) or not sf:
            raise ValueError(f'图片缺少原始文件名：{fp.name}')
        sf_stem = Path(sf).stem

        files[i][1] = fp.parent / ( sf_stem + '_0' + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                files[j][1] = fp_j.parent / ( sf_stem + ('_%d' % (c)) + fp_j.suffix )
                files[j][3] = True
                c += 1

    changes = []
    for source, destination, _, _ in files:
        changes.append({'source': source.name, 'target': destination.name})
        sidecar = source.with_suffix(source.suffix + '.landmarks.json')
        if sidecar.exists():
            changes.append({'source': sidecar.name, 'target': destination.name + '.landmarks.json'})
    receipt = execute_plan(input_path, changes, operation='recover-filenames', dry_run=dry_run)
    io.log_info(f'文件名恢复预览：{len(files)} 张。' if dry_run else f'原始文件名已恢复；回执：{receipt["receipt_path"]}')
    return receipt

def export_faceset_mask(input_dir):
    for filename in io.progress_bar_generator(pathex.get_image_paths (input_dir), "Processing"):
        filepath = Path(filename)

        if '_mask' in filepath.stem:
            continue

        mask_filepath = filepath.parent / (filepath.stem+'_mask'+filepath.suffix)

        dflimg = DFLJPG.load(filepath)

        H,W,C = dflimg.shape

        seg_ie_polys = dflimg.get_seg_ie_polys()

        if dflimg.has_xseg_mask():
            mask = dflimg.get_xseg_mask()
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
        elif seg_ie_polys.has_polys():
            mask = np.zeros ((H,W,1), dtype=np.float32)
            seg_ie_polys.overlay_mask(mask)
        else:
            raise Exception(f'no mask in file {filepath}')


        cv2_imwrite(mask_filepath, (mask*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
