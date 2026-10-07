"""Persistent global Best Training Faceset plans; independent byte-exact copies.

Features accumulate in <=500 visible windows, then one complete-inventory
selection runs. The source files, JPEG pixels, DFL metadata and checkpoints
are never rewritten. Missing/uncertain identity remains reviewable.
"""
import argparse
from contextlib import redirect_stdout, closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.best_training_faceset import (algorithm_suggestion, expression_coverage, global_training_selection,
    manual_review_decision, review_eligibility)
from core.imagelib.face_quality import face_detail_signals
from DFLIMG import DFLIMG
from facelib import LandmarksProcessor
from pose_probe_contract import probe_dataset_inventory, sha256_file
from face_quality_review import quality_review

BATCH_LIMIT = 500
PLAN_ID = re.compile(r'^plan-[a-f0-9]{32}$')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg'}


class PlanConflict(ValueError):
    pass


def _json(path, value):
    pending = path.with_name(path.name + '.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def _ordinary(path, directory=False):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        raise PlanConflict('需要普通本地路径，不能使用链接：' + str(path))
    return path


def _member(value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or PurePosixPath(value).is_absolute() or any(part in ('', '.', '..') for part in value.split('/'))):
        raise PlanConflict('数据集成员路径无效')
    return value


def _progress(stage, count, total, detail=None):
    print('DFL_PROGRESS ' + json.dumps({'stage': stage, 'current': count, 'total': total, 'detail': detail}, ensure_ascii=False), file=sys.stderr, flush=True)


def _connect(plan):
    connection = sqlite3.connect(plan / 'features.sqlite', timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _review_state(connection, metadata):
    if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_state'").fetchone():
        row = connection.execute('SELECT metadata FROM review_state WHERE singleton=1').fetchone()
        if row is None: raise PlanConflict('人工复核事务记录不完整')
        return {**metadata, **json.loads(row['metadata'])}
    return metadata


def _load(plan_directory, *, repair_review=True):
    plan = _ordinary(plan_directory, directory=True)
    if not PLAN_ID.fullmatch(plan.name):
        raise PlanConflict('全局计划编号无效')
    metadata = json.loads(_ordinary(plan / 'plan.json').read_text(encoding='utf-8'))
    if metadata.get('schemaVersion') != 1 or metadata.get('planId') != plan.name:
        raise PlanConflict('全局计划格式无效')
    _ordinary(plan / 'features.sqlite')
    # SQLite is authoritative for a review edit. A crash between the JSON
    # mirror and commit is repaired under the same cross-process write lock.
    with closing(_connect(plan)) as connection, connection:
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_state'").fetchone():
            connection.execute('BEGIN IMMEDIATE')
            # Publication/reference confirmation may have committed while this
            # reader waited for the lock. Never mirror a pre-lock state over it.
            metadata = json.loads(_ordinary(plan / 'plan.json').read_text(encoding='utf-8'))
            if metadata.get('schemaVersion') != 1 or metadata.get('planId') != plan.name:
                raise PlanConflict('全局计划格式无效')
            current = _review_state(connection, metadata)
            if repair_review and current != metadata: _json(plan / 'plan.json', current)
            metadata = current
    return plan, metadata


def _inventory(metadata):
    dataset = Path(metadata['dataset'])
    _ordinary(dataset, directory=dataset.is_dir())
    kind, fingerprint, images = probe_dataset_inventory(dataset)
    if kind != metadata['datasetKind'] or fingerprint != metadata['datasetFingerprint']:
        raise PlanConflict('源数据集已变化；保留现有计划，重新创建完整计划')
    if _sidecar_fingerprint(images) != metadata.get('sidecarFingerprint'):
        raise PlanConflict('源关键点sidecar已变化；保留现有计划，重新创建完整计划')
    return images


def _sidecar_fingerprint(images):
    digest = hashlib.sha256()
    for image in images:
        if image.path:
            path = image.path.with_suffix(image.path.suffix + '.landmarks.json')
            if path.exists():
                digest.update(image.member.encode('utf-8') + b'\0' + sha256_file(_ordinary(path)).encode('ascii') + b'\n')
    return digest.hexdigest()


def create_plan(input_path, plan_root, *, target_count=2000, confirmed_reference_members=(),
                frames_directory=None, identity_threshold=.50, quality_model='foreground_tenengrad', minimum_new_diversity=.035):
    if type(target_count) is not int or target_count < 1 or target_count > 500000:
        raise ValueError('目标数量必须在 1..500000 之间')
    if quality_model not in ('foreground_tenengrad', 'efficient-fiqa'):
        raise ValueError('只有既有细节基线或唯一 Efficient-FIQA 可用')
    if not .30 <= identity_threshold <= .85:
        raise ValueError('身份相似度阈值须在0.30..0.85之间；不是身份概率')
    if not .005 <= minimum_new_diversity <= .30:
        raise ValueError('新增多样性门槛须在0.005..0.30之间；不能设0凑数量')
    source = Path(input_path).absolute()
    _ordinary(source, directory=source.is_dir())
    kind, fingerprint, images = probe_dataset_inventory(source)
    if not images or len(images) > 500000:
        raise ValueError('完整数据集需要1..500000张aligned')
    members = {image.member for image in images}
    references = list(dict.fromkeys(_member(value) for value in confirmed_reference_members))
    if len(references) > 32 or not set(references) <= members:
        raise ValueError('请从当前数据集中确认至多32张同一目标人物参考')
    parent = Path(plan_root).absolute()
    if source.is_dir() and parent.is_relative_to(source):
        raise ValueError('计划目录不能写入源aligned内部')
    parent.mkdir(parents=True, exist_ok=True); _ordinary(parent, directory=True)
    plan = parent / ('plan-' + uuid.uuid4().hex); plan.mkdir()
    frames = _ordinary(frames_directory, directory=True) if frames_directory else None
    frame_manifest = frames / 'frames.timeline.json' if frames else None
    metadata = {'schemaVersion': 1, 'planId': plan.name, 'dataset': str(source), 'datasetKind': kind,
        'datasetFingerprint': fingerprint, 'total': len(images), 'targetCount': target_count,
        'sidecarFingerprint': _sidecar_fingerprint(images), 'minimumNewDiversity': minimum_new_diversity,
        'confirmedReferenceMembers': references, 'identityThreshold': identity_threshold,
        'identityConfirmation': 'human-selected-reference-images; no-largest-group-assumption',
        'qualityModel': quality_model, 'analysisBatchLimit': BATCH_LIMIT, 'pairedComparisonLimit': 250,
        'qualityAdmission': 'production-current' if quality_model == 'foreground_tenengrad' else 'comparison-only',
        'framesDirectory': str(frames) if frames else None,
        'frameManifestSha256': sha256_file(frame_manifest) if frame_manifest and frame_manifest.is_file() else None,
        'state': 'analyzing', 'globalSelection': False, 'createdAt': _now(),
        'parentPlanId': None, 'reviewRevision': 0, 'reviewVersion': 0}
    _json(plan / 'plan.json', metadata)
    with _connect(plan) as connection:
        connection.executescript('CREATE TABLE records (position INTEGER PRIMARY KEY, member TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL, feature TEXT, decision TEXT);')
        connection.executemany('INSERT INTO records(position,member,sha256) VALUES(?,?,?)',
                               [(index, image.member, image.sha256) for index, image in enumerate(images)])
    return inspect_plan(plan)


def inspect_plan(plan_directory, *, offset=0, limit=BATCH_LIMIT, status=None):
    plan, metadata = _load(plan_directory)
    if metadata.get('publication'):
        receipt_path = Path(metadata['publication']['receiptPath'])
        if receipt_path.is_file():
            receipt = json.loads(_ordinary(receipt_path).read_text(encoding='utf-8'))
            if receipt.get('state') == 'withdrawn': metadata = {**metadata, 'state': 'withdrawn'}
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= BATCH_LIMIT:
        raise ValueError('计划查看每批最多500张')
    if status not in (None, 'selected', 'review', 'rejected'):
        raise ValueError('查看分类必须是selected/review/rejected')
    with _connect(plan) as connection:
        completed = connection.execute('SELECT COUNT(*) FROM records WHERE feature IS NOT NULL').fetchone()[0]
        missing = connection.execute('SELECT MIN(position) FROM records WHERE feature IS NULL').fetchone()[0]
        where = " WHERE json_extract(decision,'$.status')=?" if status else ''
        values = (status, limit, offset) if status else (limit, offset)
        records = connection.execute('SELECT position,member,sha256,feature,decision FROM records' + where + ' ORDER BY position LIMIT ? OFFSET ?', values).fetchall()
        visible_total = connection.execute('SELECT COUNT(*) FROM records' + where, (status,) if status else ()).fetchone()[0]
        can_undo = bool(metadata.get('parentPlanId') and metadata['state'] == 'finalized'
            and connection.execute("SELECT 1 FROM review_operations WHERE kind='decide' AND active=1 LIMIT 1").fetchone())
        review_summary = _review_summary(connection, can_undo=can_undo)
        history = [dict(row) for row in connection.execute('''SELECT request_id AS requestId,kind,revision,
            expected_revision AS expectedRevision,json_array_length(changes) AS memberCount,undo_of AS undoOf,
            active,created_at AS createdAt FROM review_operations ORDER BY revision DESC LIMIT 20''')] if metadata.get('parentPlanId') else []
    items = []
    for row in records:
        record = json.loads(row['decision'] or row['feature']) if row['decision'] or row['feature'] else {}
        items.append({'position': row['position'], 'member': row['member'], 'sha256': row['sha256'], 'analyzed': row['feature'] is not None,
                      'status': record.get('status'), 'classification': record.get('classification'),
                      'reasons': record.get('reasons', record.get('hardReasons') or record.get('reviewReasons') or []),
                      'quality': record.get('quality'), 'coverage': record.get('coverage'), 'identity': record.get('identity'),
                      'algorithmSuggestion': algorithm_suggestion(record) if row['decision'] else None,
                      'manualDecision': record.get('manualDecision'),
                      'reviewEligibility': review_eligibility(record, metadata['confirmedReferenceMembers'],
                          identity_threshold=metadata['identityThreshold'],
                          references_conflict=(metadata.get('selection') or {}).get('identityReferencesConflict', False))})
    return {**metadata, 'planDirectory': str(plan), 'completedCount': completed,
            'globalReady': completed == metadata['total'], 'nextOffset': missing,
            'filterStatus': status,
            'selectedRange': {'start': offset + 1 if records else None, 'end': offset + len(records) if records else None,
                              'offset': offset, 'limit': limit, 'count': len(records), 'total': visible_total},
            'items': items, 'selection': metadata.get('selection'), 'publication': metadata.get('publication'),
            'parentPlanId': metadata.get('parentPlanId'), 'reviewRevision': metadata.get('reviewRevision', 0),
            'reviewVersion': metadata.get('reviewVersion', 0),
            'reviewSummary': review_summary, 'reviewHistory': history}


class AnalysisModels:
    def __init__(self, device='cpu'):
        self.device = device
        self._detector = self._landmarks = self._identity = None

    def detect(self, rgb):
        if self._detector is None:
            from vision_detectors import FaceDetector
            self._detector = FaceDetector('yolo26s-face', device=self.device)
        return self._detector.predict(rgb)

    def landmarks(self, bgr, box):
        if self._landmarks is None:
            from facelib.LandmarkCandidates import TufaExtractor
            self._landmarks = TufaExtractor(device=self.device)
        return {'points68': self._landmarks.extract(bgr, [box])[0],
                'native98': self._landmarks.audit(bgr, [box])[0]}

    def embedding(self, bgr, points68):
        if self._identity is None:
            from vision_identity import SFaceTorch
            self._identity = SFaceTorch(device=self.device)
        from role_grouping import aligned_face_input
        from vision_identity import normalized, SFACE_SHA256
        face = aligned_face_input(bgr, points68)
        return {'embedding': normalized(self._identity.embedding_bgr(face)).tolist(),
                'model': 'sface', 'sha256': SFACE_SHA256, 'alignment': 'existing-shared-five-point-112', 'runtime': 'pytorch'}


def _pixel_hash(image):
    digest = hashlib.sha256(str(image.shape).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _read(image_ref, *, metadata=True):
    raw = image_ref.path.read_bytes() if image_ref.path else image_ref.packed_sample.read_raw_file()
    if hashlib.sha256(raw).hexdigest() != image_ref.sha256:
        raise PlanConflict('源图SHA已变化：' + image_ref.member)
    try:
        pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        pixels = None
    dfl = None
    if metadata:
        try:
            dfl = DFLIMG.load(image_ref.path or Path(image_ref.name), loader_func=lambda _: raw)
        except Exception:
            # The project data-only pickle loader rejects executable metadata.
            # Such members remain opaque rejected copies instead of aborting all.
            pass
    return raw, pixels, dfl


def _metadata_hash(raw):
    from DFLIMG import DFLJPG
    try:
        chunks = DFLJPG.load_raw('opaque.jpg', loader_func=lambda _: raw).chunks
        payloads = [chunk['data'] for chunk in chunks if chunk.get('name') == 'APP15']
        return hashlib.sha256(b''.join(payloads)).hexdigest() if payloads else None
    except Exception:
        return None


def _timeline(metadata):
    directory = metadata.get('framesDirectory')
    if not directory or not metadata.get('frameManifestSha256'):
        return {}
    root = _ordinary(directory, directory=True)
    manifest = _ordinary(root / 'frames.timeline.json')
    if sha256_file(manifest) != metadata['frameManifestSha256']:
        raise PlanConflict('原始帧时间清单已变化，请重新创建计划')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('kind') != 'extracted-frames' or data.get('schemaVersion') != 1 or not re.fullmatch('[a-f0-9]{64}', data.get('source', {}).get('sha256', '')):
        raise PlanConflict('时间清单缺少有效原始素材绑定')
    records = {}
    for frame in data['frames']:
        name = _member(frame['file'])
        if '/' in name or name in records:
            raise PlanConflict('原始帧时间清单名称不唯一')
        records[name] = {**frame, 'sourceVideoSha256': data['source']['sha256'], 'manifestSha256': metadata['frameManifestSha256']}
    return records


def _iou(a, b):
    left, top = np.maximum(a[:2], b[:2]); right, bottom = np.minimum(a[2:], b[2:])
    overlap = max(0., right - left) * max(0., bottom - top)
    denominator = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return float(overlap / denominator) if denominator > 0 else 0.


def analyze_image(image_ref, metadata, models, quality_scorer=None, timeline_records=None):
    row = {'member': image_ref.member, 'sha256': image_ref.sha256, 'hardReasons': [], 'reviewReasons': [],
           'pixelSha256': None, 'metadataSha256': None, 'coverage': None, 'timeline': None, 'embedding': None}
    raw, image, dfl = _read(image_ref)
    row['pixelSha256'] = _pixel_hash(image) if image is not None else None
    row['metadataSha256'] = _metadata_hash(raw)
    if image is None or dfl is None or not dfl.has_data():
        row['hardReasons'].append('图片或DFL元数据无法读取'); return row
    if image.shape[0] != image.shape[1] or min(image.shape[:2]) < 32:
        row['hardReasons'].append('aligned画布不符合训练要求'); return row
    height, width = image.shape[:2]
    try:
        points = np.asarray(dfl.get_landmarks(), dtype=np.float64)
    except (TypeError, ValueError, OverflowError, KeyError):
        row['hardReasons'].append('既有68点关键点明显异常'); return row
    if points.shape != (68, 2) or not np.isfinite(points).all() or min(np.ptp(points, axis=0)) <= 2:
        row['hardReasons'].append('既有68点关键点明显异常'); return row
    outside = float(np.any((points < 0) | (points >= [width, height]), axis=1).mean())
    if outside > .40:
        row['hardReasons'].append('大部分关键点超出画面'); return row
    if outside > .10: row['reviewReasons'].append('部分关键点靠近或超出边缘，需要复核')
    mask = LandmarksProcessor.get_image_hull_mask(image.shape, points)[..., 0]
    try:
        detail = face_detail_signals(image, points, foreground=mask)
    except ValueError:
        row['hardReasons'].append('有效人脸区域过小或关键点退化'); return row
    row['detailBaseline'] = detail
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    foreground = gray[mask >= .5]
    row['appearance'] = {'brightness': float(foreground.mean()), 'contrast': float(foreground.std())}
    if float(np.mean((foreground < .015) | (foreground > .985))) > .97:
        row['hardReasons'].append('人脸区域极端曝光，几乎没有有效像素'); return row
    if detail['scores']['foreground_tenengrad'] < .008:
        row['hardReasons'].append('人脸细节严重缺失或严重模糊'); return row
    try:
        xseg_mask = dfl.get_xseg_mask()
    except (ValueError, TypeError, KeyError, cv2.error):
        row['hardReasons'].append('内嵌XSeg遮罩损坏，不能直接用于训练'); return row
    if quality_scorer:
        quality = quality_scorer(image, dfl.get_dict())
    else:
        from dfl_asset_tool import bounded_image_metrics
        aggregate = quality_review(image, bounded_image_metrics(image, xseg_mask), dfl,
                                   pose_estimator=LandmarksProcessor.estimate_pitch_yaw_roll)
        quality = {'model': 'foreground_tenengrad', 'score': aggregate['score'] * 100., 'minimumAcceptable': 5.,
                   'provenance': {'method': aggregate['method'], 'components': aggregate['components'],
                                  'displayScale': '100*existing aggregate; quality signal, not accuracy or identity probability',
                                  'cutoffStatus': 'uncalibrated development floor 5/100; severe diagnostics govern rejection',
                                  'sourceHash': image_ref.sha256, 'baselineMatches': 'efficient-fiqa-benchmark current aggregate'}}
    if quality.get('model') != metadata['qualityModel']:
        raise PlanConflict('当前质量评分器与计划声明不一致')
    row['quality'] = quality
    try:
        detection = models.detect(np.ascontiguousarray(image[:, :, ::-1]))
        candidates = detection['detections']
        row['detector'] = {'model': 'yolo26s-face', 'assetIdentity': detection.get('asset_identity'),
                           'inputSha256': image_ref.sha256, 'scope': 'current-aligned-pixels; not-original-source-box-accuracy',
                           'faceCount': len(candidates)}
        substantial = [item for item in candidates if item['confidence'] >= .65
                       and np.prod(np.asarray(item['box_xyxy'])[2:] - np.asarray(item['box_xyxy'])[:2]) >= width * height * .025]
        if len(substantial) > 1:
            row['hardReasons'].append('裁片内包含多个明显人脸'); return row
        if not candidates:
            row['reviewReasons'].append('现有检测器未确认人脸，可能是困难侧脸或遮挡')
            primary = None
        else:
            bounds = np.array([*points.min(axis=0), *points.max(axis=0)])
            primary = max(candidates, key=lambda item: _iou(bounds, np.asarray(item['box_xyxy'])))
            overlap = _iou(bounds, np.asarray(primary['box_xyxy']))
            row['detector'].update(bestOverlap=overlap, confidence=primary['confidence'])
            if overlap < .08 and primary['confidence'] >= .80:
                row['hardReasons'].append('清晰检测人脸与既有关键点明显不重合'); return row
            if overlap < .20: row['reviewReasons'].append('检测位置与既有关键点不够一致')
        from dfl_asset_tool import inspect_native_landmarks
        native = inspect_native_landmarks(dfl, width, height)
        if native and native.get('available'):
            native_points = native['landmarks']
            row['native98'] = {'source': 'verified-existing-native98-affine-and-canvas', 'inputSha256': image_ref.sha256, 'definition': 'WFLW98'}
        elif primary is not None:
            fresh = models.landmarks(image, primary['box_xyxy'])
            native_points = fresh['native98']['points_original']
            fresh68 = np.asarray(fresh['points68'], dtype=np.float64)
            residual = float(np.median(np.linalg.norm(fresh68 - points, axis=1)) / max(np.linalg.norm(bounds[2:] - bounds[:2]), 1))
            row['native98'] = {'source': 'new-current-aligned-TUFA68-98-diagnostic; originals-unchanged',
                               'inputSha256': image_ref.sha256, 'definition': 'WFLW98',
                               'existing68MedianResidualFraction': residual,
                               'assetIdentity': fresh['native98'].get('asset_identity')}
            if residual > .18: row['reviewReasons'].append('新TUFA与既有关键点存在明显差异，需先检查对齐')
        else:
            native_points = None
        if native_points is not None:
            angles = LandmarksProcessor.estimate_pitch_yaw_roll(points, size=width)
            row['coverage'] = expression_coverage(native_points, angles)
        else:
            row['reviewReasons'].append('缺少可靠原生98点眼口覆盖证据')
        descriptor = models.embedding(image, points)
        row['embedding'] = descriptor['embedding']
        row['identityFeature'] = {key: value for key, value in descriptor.items() if key != 'embedding'}
    except (ValueError, TypeError, RuntimeError, ImportError, KeyError, cv2.error) as error:
        row['reviewReasons'].append('检测、关键点或人脸特征尚不可验证：' + str(error))
    source_name = dfl.get_source_filename()
    row['sourceFilename'] = source_name
    record = (timeline_records or {}).get(source_name)
    if record:
        source_path = Path(metadata['framesDirectory']) / source_name
        provenance = dfl.get_dict().get('landmark_provenance', {})
        if (provenance.get('processedSourceSha256') == record.get('imageSha256') and source_path.is_file()
                and sha256_file(_ordinary(source_path)) == record['imageSha256']
                and type(record.get('pts')) is int and type(record.get('sourceFrameIndex')) is int
                and isinstance(record.get('timeBase'), list) and len(record['timeBase']) == 2
                and all(type(v) is int and v > 0 for v in record['timeBase'])):
            row['timeline'] = {key: record.get(key) for key in ('sourceVideoSha256', 'sourceFrameIndex', 'pts', 'timeBase', 'shotId', 'manifestSha256', 'imageSha256')}
        else:
            row['temporalEvidenceUnavailableReason'] = '未通过原始帧SHA/提取来源/整数PTS绑定；不猜测连续帧'
    else:
        row['temporalEvidenceUnavailableReason'] = '没有SHA绑定原始帧时间清单；仍可用像素和人脸特征选择'
    return row


def analyze_batch(plan_directory, *, offset=0, limit=BATCH_LIMIT, device='cpu', quality_scorer=None, models=None):
    plan, metadata = _load(plan_directory)
    if metadata['state'] != 'analyzing': raise PlanConflict('当前计划已经定案，不能重新覆盖特征')
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= BATCH_LIMIT:
        raise ValueError('分析范围要求offset>=0、limit在1..500')
    images = _inventory(metadata)
    batch = images[offset:offset + limit]
    if not batch: raise ValueError('指定批次为空')
    if metadata['qualityModel'] == 'efficient-fiqa' and quality_scorer is None:
        from vision_fiqa import make_quality_scorer
        quality_scorer = make_quality_scorer(device=device)
    try:
        models = models or AnalysisModels(device=device)
        timeline = _timeline(metadata)
        with _connect(plan) as connection:
            for index, image_ref in enumerate(batch):
                position = offset + index
                stored = connection.execute('SELECT member,sha256,feature FROM records WHERE position=?', (position,)).fetchone()
                if not stored or stored['member'] != image_ref.member or stored['sha256'] != image_ref.sha256:
                    raise PlanConflict('全量计划成员次序或SHA发生变化')
                if stored['feature'] is None:
                    with redirect_stdout(sys.stderr):
                        record = analyze_image(image_ref, metadata, models, quality_scorer, timeline)
                    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
                    connection.execute('UPDATE records SET feature=? WHERE position=?', (encoded, position))
                _progress('best-training-features', index + 1, len(batch), image_ref.member)
        _inventory(metadata)
        return inspect_plan(plan, offset=offset, limit=limit)
    finally:
        close = getattr(quality_scorer, "close", None)
        if close is not None:
            close()


def _revision(value):
    if type(value) is not int or value < 0:
        raise ValueError('复核版本必须是非负整数')
    return value


def _request_id(value):
    try:
        if not isinstance(value, str): raise ValueError()
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise ValueError('每次复核操作需要独立的UUID；响应不确定时使用原UUID重试')


def _references_match(metadata, values):
    references = [_member(value) for value in values]
    if len(references) > 32 or len(set(references)) != len(references):
        raise ValueError('身份参照须为至多32张不同的图像')
    if set(references) != set(metadata['confirmedReferenceMembers']):
        raise PlanConflict('身份参照已变化，请重新确认并生成选集后再复核')
    return sorted(references)


def _review_metadata_patch(metadata):
    return {key: metadata[key] for key in ('selection', 'reviewRevision', 'reviewUpdatedAt') if key in metadata}


def _review_summary(connection, *, can_undo=False):
    summary = {'pendingCount': 0, 'deferredCount': 0, 'adequateReserveCount': 0,
               'decidedCount': 0, 'manualCounts': {'keep': 0, 'exclude': 0, 'defer': 0}, 'canUndo': can_undo}
    rows = connection.execute("""SELECT
        COALESCE(json_extract(decision,'$.algorithmSuggestion.classification'),json_extract(decision,'$.classification')) AS suggestion,
        json_extract(decision,'$.manualDecision.decision') AS choice,COUNT(*) AS count
        FROM records WHERE decision IS NOT NULL GROUP BY suggestion,choice""")
    for row in rows:
        choice, count = row['choice'], row['count']
        if choice in summary['manualCounts']: summary['manualCounts'][choice] += count
        if row['suggestion'] == 'needs-review' and choice not in ('keep', 'exclude'): summary['pendingCount'] += count
        if row['suggestion'] == 'adequate-reserve' and choice is None: summary['adequateReserveCount'] += count
    summary['deferredCount'] = summary['manualCounts']['defer']
    summary['decidedCount'] = summary['manualCounts']['keep'] + summary['manualCounts']['exclude']
    return summary


def _review_selection(connection, metadata):
    counts = {status: 0 for status in ('selected', 'review', 'rejected')}
    for row in connection.execute("SELECT json_extract(decision,'$.status') AS status,COUNT(*) AS count FROM records GROUP BY status"):
        if row['status'] not in counts: raise PlanConflict('复核分类记录不完整')
        counts[row['status']] = row['count']
    if counts['selected'] > metadata['targetCount']:
        raise PlanConflict('人工保留超过数量上限；请先排除其他图，或新建更大上限的计划')
    selection = {**metadata['selection'], 'counts': counts, 'reviewRevision': metadata['reviewRevision'],
                 'manualReview': _review_summary(connection), 'algorithmCounts': metadata['selection'].get('algorithmCounts', metadata['selection']['counts'])}
    selection['selectedBuckets'] = connection.execute("SELECT COUNT(DISTINCT json_extract(decision,'$.coverage.bucket')) FROM records WHERE json_extract(decision,'$.status')='selected'").fetchone()[0]
    return selection


def create_review_plan(plan_directory, *, expected_revision, request_id):
    """Fork complete features/decisions; the original recommendation stays intact."""
    expected_revision, request_id = _revision(expected_revision), _request_id(request_id)
    parent, metadata = _load(plan_directory)
    target = parent.parent / ('plan-' + uuid.uuid5(uuid.NAMESPACE_URL, parent.name + '/review/' + request_id).hex)
    if target.exists():
        result = inspect_plan(target)
        if (result.get('parentPlanId') != parent.name or result.get('reviewCreatedByRequestId') != request_id
                or result.get('reviewCreatedFromRevision') != expected_revision):
            raise PlanConflict('同一操作UUID不能用于不同父计划或版本')
        _inventory(metadata)
        return {**result, 'reviewOperation': {'requestId': request_id, 'kind': 'create', 'reused': True}}
    with closing(_connect(parent)) as source, source:
        source.execute('BEGIN IMMEDIATE')
        metadata = _review_state(source, json.loads((parent / 'plan.json').read_text(encoding='utf-8')))
        # Two deliveries of the same request may both have observed a missing
        # target before one acquired the parent lock. Adopt that exact child.
        if target.exists():
            result = inspect_plan(target)
            if (result.get('parentPlanId') != parent.name or result.get('reviewCreatedByRequestId') != request_id
                    or result.get('reviewCreatedFromRevision') != expected_revision):
                raise PlanConflict('同一操作UUID不能用于不同父计划或版本')
            _inventory(metadata)
            return {**result, 'reviewOperation': {'requestId': request_id, 'kind': 'create', 'reused': True}}
        if metadata.get('state') not in ('finalized', 'published') or not metadata.get('globalSelection'):
            raise PlanConflict('先完成全量建议，才能创建人工复核版本')
        if metadata.get('reviewRevision', 0) != expected_revision:
            raise PlanConflict('复核版本已变化，请刷新后再创建新版')
        _inventory(metadata)
        rows = source.execute('SELECT position,member,sha256,feature,decision FROM records ORDER BY position').fetchall()
        if len(rows) != metadata['total'] or any(row['feature'] is None or row['decision'] is None for row in rows):
            raise PlanConflict('全量特征或建议不完整')
        _validate_feature_bindings([json.loads(row['feature']) for row in rows], metadata)
        child = {**metadata, 'planId': target.name, 'state': 'finalized', 'parentPlanId': parent.name,
                 'reviewRootPlanId': metadata.get('reviewRootPlanId', parent.name), 'reviewRevision': 0,
                 'reviewVersion': metadata.get('reviewVersion', 0) + 1, 'createdAt': _now(),
                 'reviewCreatedByRequestId': request_id, 'reviewCreatedFromRevision': expected_revision}
        child.pop('publication', None)
        child.pop('reviewUpdatedAt', None)
        staging = parent.parent / ('.pending-review-' + uuid.uuid4().hex)
        staging.mkdir()
        # A pending directory has no plan ID and cannot be consumed by WebUI.
        # It is retained on an interrupted write; no user media is held here.
        with closing(_connect(staging)) as destination, destination:
            destination.executescript('''CREATE TABLE records (position INTEGER PRIMARY KEY, member TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL, feature TEXT, decision TEXT);
                CREATE TABLE review_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), metadata TEXT NOT NULL);
                CREATE TABLE review_operations (request_id TEXT PRIMARY KEY, kind TEXT NOT NULL, expected_revision INTEGER NOT NULL,
                    revision INTEGER NOT NULL, payload_sha256 TEXT NOT NULL, changes TEXT NOT NULL, undo_of TEXT, active INTEGER NOT NULL, created_at TEXT NOT NULL);''')
            destination.executemany('INSERT INTO records VALUES(?,?,?,?,?)',
                [(row['position'], row['member'], row['sha256'], row['feature'], json.dumps({**json.loads(row['decision']),
                    'algorithmSuggestion': algorithm_suggestion(json.loads(row['decision']))}, ensure_ascii=False, allow_nan=False)) for row in rows])
            child['selection'] = _review_selection(destination, child)
            destination.execute('INSERT INTO review_state VALUES(1,?)', (json.dumps(_review_metadata_patch(child), ensure_ascii=False, allow_nan=False),))
        _inventory(metadata)
        _json(staging / 'plan.json', child)
        if target.exists(): raise PlanConflict('复核版本已存在，请使用原UUID重试读取')
        staging.rename(target)
    return {**inspect_plan(target), 'reviewOperation': {'requestId': request_id, 'kind': 'create', 'reused': False}}


def _check_keep_input(image_ref):
    raw, image, dfl = _read(image_ref)
    if image is None or dfl is None or not dfl.has_data() or image.shape[0] != image.shape[1] or min(image.shape[:2]) < 32:
        raise PlanConflict('无法读取或没有有效DFL元数据的素材不能人工保留：' + image_ref.member)
    try:
        points = np.asarray(dfl.get_landmarks(), dtype=np.float64)
        if points.shape != (68, 2) or not np.isfinite(points).all() or min(np.ptp(points, axis=0)) <= 2:
            raise ValueError('关键点异常')
        dfl.get_xseg_mask()
    except (ValueError, TypeError, KeyError, cv2.error):
        raise PlanConflict('关键点或内嵌遮罩无效，不能人工保留：' + image_ref.member)


def _review_mutation(plan_directory, *, expected_revision, request_id, expected_references, decision=None, members=(), undo=False):
    expected_revision, request_id = _revision(expected_revision), _request_id(request_id)
    names = [_member(value) for value in members]
    if not undo and (not 1 <= len(names) <= BATCH_LIMIT or len(set(names)) != len(names)):
        raise ValueError('每次人工复核需要1..500张不同成员')
    if not undo and decision not in ('keep', 'exclude', 'defer'):
        raise ValueError('人工决定须为keep、exclude或defer')
    plan, metadata = _load(plan_directory)
    kind = 'undo' if undo else 'decide'
    with closing(_connect(plan)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        metadata = _review_state(connection, json.loads((plan / 'plan.json').read_text(encoding='utf-8')))
        if not metadata.get('parentPlanId'):
            raise PlanConflict('算法推荐和已发布选集不就地改写，请先创建人工复核新版')
        references = _references_match(metadata, expected_references)
        payload = {'kind': kind, 'expectedRevision': expected_revision, 'references': references,
                   'decision': decision, 'members': sorted(names)}
        payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        previous = connection.execute('SELECT * FROM review_operations WHERE request_id=?', (request_id,)).fetchone()
        if previous:
            if previous['payload_sha256'] != payload_sha: raise PlanConflict('同一操作UUID不能用于不同的复核决定')
            operation = {'requestId': request_id, 'kind': kind, 'revision': previous['revision'], 'reused': True}
        else:
            if metadata['state'] != 'finalized': raise PlanConflict('已发布或撤回的选集不能改写，请创建复核新版')
            if metadata['reviewRevision'] != expected_revision: raise PlanConflict('复核版本已变化，请刷新后重试')
            images = {image.member: image for image in _inventory(metadata)}
            revision, created_at = expected_revision + 1, _now()
            changes, undo_of = [], None
            if undo:
                latest = connection.execute("SELECT * FROM review_operations WHERE kind='decide' AND active=1 ORDER BY revision DESC LIMIT 1").fetchone()
                if latest is None: raise PlanConflict('当前没有可撤销的人工决定')
                undo_of = latest['request_id']
                for change in json.loads(latest['changes']):
                    current = connection.execute('SELECT decision FROM records WHERE member=?', (change['member'],)).fetchone()
                    if current is None or json.loads(current['decision']) != change['after']:
                        raise PlanConflict('复核撤销记录与当前决定不一致')
                    changes.append({'member': change['member'], 'before': change['after'], 'after': change['before']})
                connection.execute('UPDATE review_operations SET active=0 WHERE request_id=?', (undo_of,))
            else:
                for name in names:
                    image = images.get(name)
                    if image is None: raise PlanConflict('人工决定含当前完整选集外的成员')
                    stored = connection.execute('SELECT decision FROM records WHERE member=?', (name,)).fetchone()
                    if stored is None or stored['decision'] is None: raise PlanConflict('成员没有完整算法建议')
                    old = json.loads(stored['decision'])
                    eligibility = review_eligibility(old, metadata['confirmedReferenceMembers'], identity_threshold=metadata['identityThreshold'],
                        references_conflict=metadata['selection'].get('identityReferencesConflict', False))
                    if decision == 'keep':
                        if not eligibility['canKeep']: raise PlanConflict('不能人工保留 ' + name + '：' + '；'.join(eligibility['blockedReasons']))
                        _check_keep_input(image)
                    changes.append({'member': name, 'before': old, 'after': manual_review_decision(old, decision,
                        revision=revision, request_id=request_id, created_at=created_at)})
            _validate_feature_bindings([change['after'] for change in changes], metadata)
            connection.executemany('UPDATE records SET decision=? WHERE member=?',
                [(json.dumps(change['after'], ensure_ascii=False, allow_nan=False), change['member']) for change in changes])
            connection.execute('INSERT INTO review_operations VALUES(?,?,?,?,?,?,?,?,?)',
                (request_id, kind, expected_revision, revision, payload_sha, json.dumps(changes, ensure_ascii=False, allow_nan=False),
                 undo_of, 0 if undo else 1, created_at))
            metadata.update(reviewRevision=revision, reviewUpdatedAt=created_at)
            metadata['selection'] = _review_selection(connection, metadata)
            connection.execute('UPDATE review_state SET metadata=? WHERE singleton=1', (json.dumps(_review_metadata_patch(metadata), ensure_ascii=False, allow_nan=False),))
            _inventory(metadata)
            # The mirror is written while SQLite owns the cross-process lock.
            # Any failure rolls back every changed member and journal entry.
            _json(plan / 'plan.json', metadata)
            operation = {'requestId': request_id, 'kind': kind, 'revision': revision, 'reused': False, 'undoOf': undo_of}
    return {**inspect_plan(plan), 'reviewOperation': operation}


def apply_review_decision(plan_directory, *, expected_revision, request_id, expected_references, decision, members):
    return _review_mutation(plan_directory, expected_revision=expected_revision, request_id=request_id,
        expected_references=expected_references, decision=decision, members=members)


def undo_review_decision(plan_directory, *, expected_revision, request_id, expected_references):
    return _review_mutation(plan_directory, expected_revision=expected_revision, request_id=request_id,
        expected_references=expected_references, undo=True)


def confirm_references(plan_directory, members):
    plan, metadata = _load(plan_directory)
    if metadata.get('parentPlanId'): raise PlanConflict('人工复核版本不能重新定义身份，请基于新参照创建算法计划')
    if metadata.get('state') == 'published': raise PlanConflict('已发布计划不能改身份参考，请创建新计划')
    images = _inventory(metadata)
    references = list(dict.fromkeys(_member(value) for value in members))
    if len(references) > 32 or not set(references) <= {image.member for image in images}:
        raise ValueError('参考图必须是当前数据集至多32张同一人物图像')
    metadata.update(confirmedReferenceMembers=references, state='analyzing', globalSelection=False)
    metadata.pop('selection', None)
    with _connect(plan) as connection: connection.execute('UPDATE records SET decision=NULL')
    _json(plan / 'plan.json', metadata)
    return inspect_plan(plan)


def finalize_plan(plan_directory):
    plan, metadata = _load(plan_directory)
    if metadata.get('parentPlanId'): raise PlanConflict('人工复核版本保留原算法建议，请使用保留、排除、暂缓或另建算法计划')
    if metadata.get('state') == 'published': return inspect_plan(plan)
    _inventory(metadata)
    with _connect(plan) as connection:
        missing = connection.execute('SELECT MIN(position) FROM records WHERE feature IS NULL').fetchone()[0]
        if missing is not None: raise PlanConflict(f'全量分析尚未完成；下一批offset={missing}，不能把单批结果当全局选择')
        records = [json.loads(row['feature']) for row in connection.execute('SELECT feature FROM records ORDER BY position')]
        _validate_feature_bindings(records, metadata)
        result = global_training_selection(records, metadata['targetCount'], metadata['confirmedReferenceMembers'],
                                           identity_threshold=metadata['identityThreshold'], minimum_new_diversity=metadata['minimumNewDiversity'])
        connection.executemany('UPDATE records SET decision=? WHERE member=?',
            [(json.dumps({key: value for key, value in row.items() if key != 'embedding'}, ensure_ascii=False, allow_nan=False), row['member']) for row in result['records']])
    metadata.update(state='finalized', globalSelection=True, selection={key: value for key, value in result.items() if key != 'records'})
    _json(plan / 'plan.json', metadata)
    return inspect_plan(plan)


def _validate_feature_bindings(records, metadata):
    _timeline(metadata)
    checked = set()
    for record in records:
        timeline = record.get('timeline')
        if timeline:
            key = (timeline['sourceFrameIndex'], timeline['imageSha256'])
            if key in checked: continue
            checked.add(key)
            source = Path(metadata['framesDirectory']) / record['sourceFilename']
            if not source.is_file() or sha256_file(_ordinary(source)) != timeline['imageSha256']:
                raise PlanConflict('原始帧SHA已变化；拒绝使用过期连续帧证据')


def verify_source(plan_directory):
    plan, metadata = _load(plan_directory, repair_review=False)
    _inventory(metadata)
    with closing(_connect(plan)) as connection:
        _validate_feature_bindings((json.loads(row['feature']) for row in connection.execute('SELECT feature FROM records WHERE feature IS NOT NULL')), metadata)
    return {'schemaVersion': 1, 'planId': plan.name, 'sourceValid': True,
            'sourceFingerprint': metadata['datasetFingerprint'], 'sidecarFingerprint': metadata['sidecarFingerprint'],
            'reviewRevision': metadata.get('reviewRevision', 0)}


def publish_plan(plan_directory, output_root, *, dry_run=False, expected_revision=None):
    plan, metadata = _load(plan_directory)
    if not metadata.get('parentPlanId'):
        return _publish_plan(plan, output_root, dry_run=dry_run, loaded=(plan, metadata))
    if expected_revision is None: raise PlanConflict('发布人工复核版本须确认当前版本，避免使用过期决定')
    expected_revision = _revision(expected_revision)
    with closing(_connect(plan)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        metadata = _review_state(connection, json.loads((plan / 'plan.json').read_text(encoding='utf-8')))
        if metadata['reviewRevision'] != expected_revision: raise PlanConflict('复核版本已变化，请重新预演并确认发布')
        # No manual edit can commit while copies/receipt are being published.
        return _publish_plan(plan, output_root, dry_run=dry_run, loaded=(plan, metadata))


def _publish_plan(plan_directory, output_root, *, dry_run=False, loaded=None):
    plan, metadata = loaded or _load(plan_directory)
    if metadata['qualityModel'] != 'foreground_tenengrad':
        raise PlanConflict('Efficient-FIQA仅供评测比较，未通过收益门槛，不能发布生产训练子集')
    if metadata['state'] not in ('finalized', 'published') or not metadata.get('globalSelection'):
        raise PlanConflict('必须完成全量分析和全局选择后再生成训练子集')
    if metadata['state'] == 'published':
        return _reuse_publication(Path(metadata['publication']['outputDirectory']), plan, metadata)
    images = _inventory(metadata)
    parent = Path(output_root).absolute()
    if Path(metadata['dataset']).is_dir() and parent.is_relative_to(Path(metadata['dataset'])):
        raise ValueError('新训练批次不能写入源aligned内部')
    if not dry_run: parent.mkdir(parents=True, exist_ok=True)
    if parent.exists(): _ordinary(parent, directory=True)
    identifier = 'best-training-' + plan.name.removeprefix('plan-')
    target = parent / identifier
    if target.exists():
        publication = _reuse_publication(target, plan, metadata)
        if not dry_run:
            metadata.update(state='published', publication={key: value for key, value in publication.items() if key != 'reused'})
            _json(plan / 'plan.json', metadata)
        return {**publication, 'dryRun': dry_run}
    with _connect(plan) as connection:
        decisions = {row['member']: json.loads(row['decision']) for row in connection.execute('SELECT member,decision FROM records')}
    _validate_feature_bindings(list(decisions.values()), metadata)
    if set(decisions) != {image.member for image in images}:
        raise PlanConflict('全局选择记录不完整')
    if dry_run:
        return {'schemaVersion': 1, 'dryRun': True, 'planId': plan.name, 'outputDirectory': str(target),
                'counts': metadata['selection']['counts'], 'sourceFingerprint': metadata['datasetFingerprint'],
                'allOriginalsRetained': True, 'independentCopies': True, 'receiptPath': None}
    staging = parent / ('.pending-' + identifier + '-' + uuid.uuid4().hex); staging.mkdir()
    receipt = {'schemaVersion': 1, 'kind': 'best-training-faceset', 'batchId': identifier, 'planId': plan.name,
        'state': 'preparing', 'source': {'dataset': metadata['dataset'], 'kind': metadata['datasetKind'], 'fingerprint': metadata['datasetFingerprint']},
        'selection': metadata['selection'], 'counts': metadata['selection']['counts'], 'entries': [],
        'allOriginalsRetained': True, 'independentCopies': True, 'originalPixelsAndMetadataPreserved': True,
        'newTraining': False}
    if metadata.get('parentPlanId'):
        receipt.update(parentPlanId=metadata['parentPlanId'], reviewVersion=metadata['reviewVersion'],
                       reviewRevision=metadata['reviewRevision'], reviewRootPlanId=metadata['reviewRootPlanId'])
    try:
        for status in ('selected', 'review', 'rejected'): (staging / status).mkdir()
        for index, image_ref in enumerate(images):
            decision = decisions[image_ref.member]
            status = decision.get('status')
            if status not in ('selected', 'review', 'rejected'): raise PlanConflict('全局选择类别无效')
            raw, image, _ = _read(image_ref, metadata=False)
            destination = staging / status / _member(image_ref.member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream: stream.write(raw)
            if sha256_file(destination) != image_ref.sha256: raise PlanConflict('独立副本字节校验失败')
            pixel_sha = _pixel_hash(image) if image is not None else None
            if pixel_sha != decision['pixelSha256']: raise PlanConflict('计划像素证据已变化')
            if _metadata_hash(raw) != decision['metadataSha256']: raise PlanConflict('计划APP15证据已变化')
            entry = {'member': image_ref.member, 'destination': f'{status}/{image_ref.member}', 'sha256': image_ref.sha256,
                     'pixelSha256': pixel_sha, 'metadataSha256': decision['metadataSha256'], 'status': status,
                     'classification': decision['classification'], 'reasons': decision['reasons'][:3],
                     'quality': decision.get('quality'), 'coverage': decision.get('coverage'), 'duplicateOf': decision.get('duplicateOf')}
            if metadata.get('parentPlanId'):
                entry.update(algorithmSuggestion=algorithm_suggestion(decision), manualDecision=decision.get('manualDecision'))
            if image_ref.path:
                sidecar = image_ref.path.with_suffix(image_ref.path.suffix + '.landmarks.json')
                if sidecar.is_file():
                    _ordinary(sidecar)
                    sidecar_sha = sha256_file(sidecar)
                    copied_sidecar = destination.with_suffix(destination.suffix + '.landmarks.json')
                    shutil.copyfile(sidecar, copied_sidecar)
                    if sha256_file(sidecar) != sidecar_sha or sha256_file(copied_sidecar) != sidecar_sha:
                        raise PlanConflict('关键点sidecar复制时发生变化或副本校验失败')
                    entry['sidecarSha256'] = sidecar_sha
            receipt['entries'].append(entry)
            _progress('best-training-publish', index + 1, len(images), image_ref.member)
        _inventory(metadata)
        receipt['state'] = 'committed'
        _json(staging / 'receipt.json', receipt)
        # Do not replace existing directories on any platform.
        if target.exists(): raise PlanConflict('目标批次已存在，保留暂存副本')
        staging.rename(target)
        publication = {'schemaVersion': 1, 'outputDirectory': str(target), 'receiptPath': str(target / 'receipt.json'),
                       'counts': receipt['counts'], 'allOriginalsRetained': True, 'independentCopies': True}
        metadata.update(state='published', publication=publication)
        _json(plan / 'plan.json', metadata)
        return publication
    except BaseException as error:
        receipt.update(state='failed-before-publication', error=str(error))
        if staging.exists(): _json(staging / 'receipt.json', receipt)
        raise


def _reuse_publication(target, plan, metadata):
    target = _ordinary(target, directory=True)
    receipt_path = _ordinary(target / 'receipt.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('kind') != 'best-training-faceset' or receipt.get('planId') != plan.name
            or receipt.get('state') != 'committed' or receipt.get('selection') != metadata['selection']
            or receipt.get('source', {}).get('fingerprint') != metadata['datasetFingerprint']):
        raise PlanConflict('现有批次不匹配当前完整计划或已撤回，不覆盖任何副本')
    with _connect(plan) as connection:
        expected = {row['member']: json.loads(row['decision']) for row in connection.execute('SELECT member,decision FROM records')}
    entries = receipt.get('entries', [])
    if len(entries) != len(expected) or len({entry['member'] for entry in entries}) != len(expected):
        raise PlanConflict('现有批次回执成员不完整')
    for entry in entries:
        old = expected.get(entry['member'])
        if (old is None or any(entry.get(key) != old.get(key) for key in ('sha256', 'pixelSha256', 'metadataSha256', 'status'))
                or entry.get('destination') != old['status'] + '/' + entry['member']):
            raise PlanConflict('现有批次回执与计划SHA或分类不匹配')
        if metadata.get('parentPlanId') and (entry.get('algorithmSuggestion') != algorithm_suggestion(old)
                or entry.get('manualDecision') != old.get('manualDecision')):
            raise PlanConflict('现有批次人工决定或原算法建议与计划不匹配')
    checked = recover_publication(receipt_path, dry_run=True)
    if checked.get('conflicts'): raise PlanConflict('; '.join(checked['conflicts']))
    return {'schemaVersion': 1, 'outputDirectory': str(target), 'receiptPath': str(receipt_path),
            'counts': receipt['counts'], 'allOriginalsRetained': True, 'independentCopies': True, 'reused': True}


def recover_publication(receipt_path, *, dry_run=False):
    receipt_path = _ordinary(receipt_path)
    batch = _ordinary(receipt_path.parent, directory=True)
    if receipt_path.name != 'receipt.json' or not re.fullmatch('best-training-[a-f0-9]{32}', batch.name):
        raise PlanConflict('训练批次回执位置无效')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('kind') != 'best-training-faceset' or receipt.get('batchId') != batch.name:
        raise PlanConflict('训练批次回执无效')
    if receipt['state'] == 'withdrawn': return {'receiptPath': str(receipt_path), 'state': 'withdrawn', 'reused': True}
    if receipt['state'] != 'committed': raise PlanConflict('该批次尚未完成发布，原件始终保留')
    conflicts = []
    expected = set()
    for entry in receipt['entries']:
        relative = _member(entry['destination']); expected.add(relative)
        destination = batch / relative
        if not destination.is_file() or sha256_file(_ordinary(destination)) != entry['sha256']:
            conflicts.append('副本已修改或缺失：' + relative)
        if entry.get('sidecarSha256'):
            sidecar = destination.with_suffix(destination.suffix + '.landmarks.json')
            expected.add(relative + '.landmarks.json')
            if not sidecar.is_file() or sha256_file(_ordinary(sidecar)) != entry['sidecarSha256']:
                conflicts.append('sidecar已修改或缺失：' + relative)
    actual = {p.relative_to(batch).as_posix() for status in ('selected', 'review', 'rejected') for p in (batch / status).rglob('*') if p.is_file()}
    if actual != expected: conflicts.append('发布目录含新增或缺失文件')
    if dry_run: return {'receiptPath': str(receipt_path), 'dryRun': True, 'conflicts': conflicts, 'allOriginalsRetained': True}
    if conflicts: raise PlanConflict('; '.join(conflicts))
    archive = batch / 'withdrawn'; archive.mkdir()
    moved = []
    try:
        for status in ('selected', 'review', 'rejected'):
            (batch / status).rename(archive / status); moved.append(status)
        receipt['state'] = 'withdrawn'; _json(receipt_path, receipt)
    except BaseException:
        for status in reversed(moved): (archive / status).rename(batch / status)
        if archive.exists() and not any(archive.iterdir()): archive.rmdir()
        raise
    return {'receiptPath': str(receipt_path), 'state': 'withdrawn', 'withdrawnDirectory': str(archive),
            'allOriginalsRetained': True, 'allCopiesRetained': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status')
    init = commands.add_parser('init'); init.add_argument('--input', type=Path, required=True); init.add_argument('--plan-root', type=Path, required=True)
    init.add_argument('--target-count', type=int, default=2000); init.add_argument('--reference', action='append', default=[])
    init.add_argument('--frames', type=Path); init.add_argument('--identity-threshold', type=float, default=.50)
    init.add_argument('--quality-model', choices=('foreground_tenengrad', 'efficient-fiqa'), default='foreground_tenengrad')
    init.add_argument('--minimum-diversity', type=float, default=.035)
    for action in ('inspect', 'analyze', 'finalize', 'confirm', 'publish', 'verify-source', 'review-create', 'review-decide', 'review-undo'):
        sub = commands.add_parser(action); sub.add_argument('--plan', type=Path, required=True)
        if action in ('inspect', 'analyze'): sub.add_argument('--offset', type=int, default=0); sub.add_argument('--limit', type=int, default=BATCH_LIMIT)
        if action == 'inspect': sub.add_argument('--status', choices=('selected', 'review', 'rejected'))
        if action == 'analyze': sub.add_argument('--device', default='cpu')
        if action == 'confirm': sub.add_argument('--reference', action='append', default=[])
        if action == 'publish':
            sub.add_argument('--output-root', type=Path, required=True); sub.add_argument('--dry-run', action='store_true')
            sub.add_argument('--expected-revision', type=int)
        if action in ('review-create', 'review-decide', 'review-undo'):
            sub.add_argument('--expected-revision', type=int, required=True)
            sub.add_argument('--request-id', required=True)
        if action in ('review-decide', 'review-undo'): sub.add_argument('--reference', action='append', default=[])
        if action == 'review-decide':
            sub.add_argument('--decision', choices=('keep', 'exclude', 'defer'), required=True)
            sub.add_argument('--member', action='append', default=[])
    recover = commands.add_parser('recover'); recover.add_argument('--receipt', type=Path, required=True); recover.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        if args.command == 'status':
            try:
                from vision_fiqa import availability
                efficient = availability()
            except (ImportError, OSError, ValueError) as error:
                efficient = {'available': False, 'model': 'efficient-fiqa', 'admission': 'comparison-only', 'default': False, 'reason': str(error)}
            result = {'schemaVersion': 1, 'qualityDefault': 'foreground_tenengrad', 'efficientFiqa': efficient,
                      'analysisBatchLimit': BATCH_LIMIT, 'pairedComparisonLimit': 250}
        elif args.command == 'init': result = create_plan(args.input, args.plan_root, target_count=args.target_count, confirmed_reference_members=args.reference,
            frames_directory=args.frames, identity_threshold=args.identity_threshold, quality_model=args.quality_model, minimum_new_diversity=args.minimum_diversity)
        elif args.command == 'inspect': result = inspect_plan(args.plan, offset=args.offset, limit=args.limit, status=args.status)
        elif args.command == 'analyze': result = analyze_batch(args.plan, offset=args.offset, limit=args.limit, device=args.device)
        elif args.command == 'confirm': result = confirm_references(args.plan, args.reference)
        elif args.command == 'finalize': result = finalize_plan(args.plan)
        elif args.command == 'verify-source': result = verify_source(args.plan)
        elif args.command == 'review-create': result = create_review_plan(args.plan, expected_revision=args.expected_revision, request_id=args.request_id)
        elif args.command == 'review-decide': result = apply_review_decision(args.plan, expected_revision=args.expected_revision, request_id=args.request_id,
            expected_references=args.reference, decision=args.decision, members=args.member)
        elif args.command == 'review-undo': result = undo_review_decision(args.plan, expected_revision=args.expected_revision, request_id=args.request_id,
            expected_references=args.reference)
        elif args.command == 'publish': result = publish_plan(args.plan, args.output_root, dry_run=args.dry_run, expected_revision=args.expected_revision)
        else: result = recover_publication(args.receipt, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
