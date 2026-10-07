import math
import multiprocessing
import json
import os
import re
import hashlib
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, InteractiveMergerSubprocessor, MergerConfig
from merger.quality_merge import load_reviewed_masks, sha256
from merger.temporal_geometry import bind_timeline, stabilize_geometry, hard_cut_thumbnail
from core.media_timeline import write_json


def apply_web_merge_config(cfg, raw):
    """Apply the guided WebUI controls without opening native input windows."""
    values = json.loads(raw)
    if not isinstance(values, dict) or cfg.type != MergerConfig.TYPE_MASKED:
        raise ValueError('DFL_WEB_MERGE_CONFIG requires a masked merge configuration object')
    modes = {'original', 'overlay', 'hist-match', 'seamless', 'seamless-hist-match', 'raw-rgb', 'raw-predict'}
    color_modes = {'none': 0, 'rct': 1, 'lct': 2, 'mkl': 3, 'mkl-m': 4,
                   'idt': 5, 'idt-m': 6, 'sot-m': 7, 'mix-m': 8, 'robust-lab': 9, 'lab-quantile': 10}
    bounds = {
        'maskMode': ('mask_mode', 4, 1, 11),
        'erodeMask': ('erode_mask_modifier', 0, -400, 400),
        'blurMask': ('blur_mask_modifier', 0, 0, 400),
        'motionBlur': ('motion_blur_power', 0, 0, 100),
        'faceScale': ('output_face_scale', 0, -50, 50),
        'sharpenMode': ('sharpen_mode', 0, 0, 2),
        'sharpenAmount': ('blursharpen_amount', 0, -100, 100),
        'superResolution': ('super_resolution_power', 0, 0, 100),
        'imageDenoise': ('image_denoise_power', 0, 0, 500),
        'bicubicDegrade': ('bicubic_degrade_power', 0, 0, 100),
        'colorDegrade': ('color_degrade_power', 0, 0, 100),
        'randomSeed': ('random_seed', 0, 0, 4294967295),
        'geometryStrength': ('geometry_strength', 0, 0, 100),
        'blendWidth': ('blend_width', 3, 0, 20),
        'workers': (None, min(8, multiprocessing.cpu_count()), 1, 128),
    }
    unknown = set(values) - set(bounds) - {'mode', 'colorTransfer', 'geometryMode', 'blendMode', 'superResolutionModel'}
    if unknown:
        raise ValueError('Unknown Web merge parameters: ' + ', '.join(sorted(unknown)))
    mode, color = values.get('mode', 'overlay'), values.get('colorTransfer', 'none')
    if mode not in modes or color not in color_modes:
        raise ValueError('Unsupported Web merge mode or color transfer')
    parsed = {}
    for key, (attribute, default, minimum, maximum) in bounds.items():
        value = values.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or not minimum <= value <= maximum:
            raise ValueError(f'Invalid Web merge parameter {key}: {value!r}')
        parsed[key] = int(value)
    geometry = values.get('geometryMode', 'off')
    blend = values.get('blendMode', 'legacy')
    if geometry not in ('off', 'bounded-center-scale') or blend not in ('legacy', 'distance', 'multiband'):
        raise ValueError('Unsupported geometry or blend candidate')
    enhancer = values.get('superResolutionModel', 'mambairv2')
    if enhancer not in ('mambairv2', 'realesrgan-x4plus'):
        raise ValueError('Unsupported predicted-face enhancement model')
    cfg.super_resolution_model = enhancer
    cfg.geometry_mode, cfg.blend_mode = geometry, blend
    cfg.mode, cfg.color_transfer_mode = mode, color_modes[color]
    for key, (attribute, *_bounds) in bounds.items():
        if attribute is not None:
            setattr(cfg, attribute, parsed[key])
    return min(parsed['workers'], multiprocessing.cpu_count())


def read_packed_alignment(sample):
    """Bind decoded metadata to the exact member bytes, not a same-named file."""
    filepath = Path(sample.filename)
    if sample.person_name is not None:
        filepath = Path(sample.person_name) / filepath
    payload = sample.read_raw_file()
    return (filepath, DFLIMG.load(filepath, loader_func=lambda _path: payload),
            hashlib.sha256(payload).hexdigest())


def alignment_identity(entry, packed_identities, reviewed):
    filepath = Path(entry[1])
    packed_sha = packed_identities.get(filepath.as_posix())
    digest = packed_sha if packed_sha is not None else sha256(filepath) if filepath.is_file() else None
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('Alignment content SHA256 required')
    points = np.asarray(entry[0], dtype='<f8')
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError('Finite original source 68 landmarks required')
    return {'file': filepath.name, 'sha256': digest,
            'storage': 'packed' if packed_sha is not None or not filepath.is_file() else 'plain',
            'packedMember': filepath.as_posix() if packed_sha is not None else None,
            'sourceFilename': Path(entry[2]).name,
            'sourceLandmarksSha256': hashlib.sha256(points.tobytes()).hexdigest(),
            'reviewedMask': reviewed}


def prepare_merge_frames(paths, alignments, cfg, aligned_path, packed_identities=None):
    """Preflight precedes the subprocess clearing/publishing output images."""
    if not paths:
        raise ValueError("No merge input frames")
    timings, timing_audit = bind_timeline(paths)
    packed_identities = packed_identities or {}
    required = [entry[1].name for path in paths for entry in alignments.get(Path(path).stem, [])]
    reviewed = load_reviewed_masks(aligned_path, Path(paths[0]).parent, required) if cfg.mask_mode in (10, 11) else {}
    frames, records, previous_image = [], [], None
    audit = {'schemaVersion': 1, 'kind': 'merge-quality-audit', 'status': 'prepared',
             'timeline': timing_audit, 'configuration': cfg.get_config(),
             'seedPolicy': 'local per-track seed: (randomSeed + 104729 * trackId) modulo 2**32; no global RNG mutation',
             'geometryPolicy': 'bounded center/scale affine only; original 68 landmarks unchanged',
             'hardCutGuard': 'supplemental 48x48 mean absolute RGB difference >0.30; conservative guard, not ground truth',
             'frames': []}
    for path, timing in zip(paths, timings):
        path = Path(path)
        entries = alignments.get(path.stem, [])
        points = [entry[0] for entry in entries]
        image_hash = sha256(path)
        fi = FrameInfo(filepath=path, landmarks_list=points,
                       reviewed_masks=[reviewed.get(entry[1].name) for entry in entries],
                       timing=timing, source_sha256=image_hash)
        frames.append(InteractiveMergerSubprocessor.Frame(frame_info=fi))
        item = {'file': path.name, 'sourceSha256': image_hash,
                'alignments': [alignment_identity(entry, packed_identities, cfg.mask_mode in (10, 11)) for entry in entries],
                'timing': timing, 'geometry': []}
        audit['frames'].append(item)
        for alignment in item['alignments']:
            if alignment['file'] in reviewed:
                record = reviewed[alignment['file']]
                alignment['reviewedReceiptSha256'] = record['receiptSha256']
                alignment['reviewedSourceFrameSha256'] = record['sourceSha256']
                alignment['sourceToAlignedAffine'] = record['sourceToAlignedAffine'].tolist()
                alignment['maskCanvasWh'] = list(reversed(record['mask'].shape))
        fi.aligned_identities = item['alignments']
        if timing is None:
            continue
        image = cv2_imread(path)
        if image is None:
            raise ValueError('Cannot decode merge source: ' + path.name)
        record = {**timing, 'faces': [], 'cutBefore': bool(timing.get('cutBefore'))}
        if previous_image is not None and hard_cut_thumbnail(previous_image, image):
            record['cutBefore'] = True
            item['supplementalHardCut'] = True
        previous_image = image
        for landmarks in points:
            landmarks = np.asarray(landmarks)
            if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
                raise ValueError('Finite original source 68 landmarks required')
            matrix = LandmarksProcessor.get_transform_mat(landmarks, 256, face_type=FaceType.FULL)
            corners = LandmarksProcessor.transform_points([(127, 127), (127, 0)], matrix, True)
            record['faces'].append({'center': corners[0].tolist(), 'scale': float(np.linalg.norm(corners[1] - corners[0]) * 2)})
        records.append(record)
    if records:
        strength = cfg.geometry_strength if cfg.geometry_mode == 'bounded-center-scale' else 0
        stabilize_geometry(records, strength)
        for frame, item, record in zip(frames, audit['frames'], records):
            frame.frame_info.geometry = record['faces']
            item['geometry'] = record['faces']
            for face in item['geometry']:
                track = face.get('trackId')
                face['randomSeed'] = (cfg.random_seed + 104729 * (track if track is not None else item['geometry'].index(face))) % 4294967296
    elif cfg.geometry_mode != 'off':
        raise ValueError('Geometry stabilization requires a SHA-bound integer PTS timeline; re-extract original frames')
    return frames, audit


def select_preview_frames(paths, frame_start, frame_count):
    """A bounded, explicit 1-based window, without silent truncation."""
    if isinstance(frame_start, bool) or not isinstance(frame_start, int) or frame_start < 1:
        raise ValueError('Preview frame start must be a positive integer')
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 1 <= frame_count <= 20:
        raise ValueError('Preview frame count must be an integer from 1 to 20')
    if frame_start - 1 + frame_count > len(paths):
        raise ValueError(f'Preview range {frame_start}–{frame_start + frame_count - 1} exceeds {len(paths)} source frames')
    return paths[frame_start - 1:frame_start - 1 + frame_count]


def validate_preview_output_paths(input_path, output_path, output_mask_path):
    """Validate before mkdir, model loading, or any merger output clearing."""
    input_path, output_path, output_mask_path = map(Path, (input_path, output_path, output_mask_path))
    if input_path.name != 'data_dst':
        raise ValueError('Preview input must be the project data_dst directory')
    workspace = input_path.resolve().parent
    preview_root = workspace / '.webui' / 'merge-previews'
    directory = output_path.parent
    if (output_path.name != 'merged' or output_mask_path.name != 'merged_mask'
            or output_mask_path.parent.resolve() != directory.resolve()
            or directory.parent.resolve() != preview_root.resolve()
            or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9-]{5,63}', directory.name)):
        raise ValueError('Preview outputs must be an independent managed merge-previews directory')
    for candidate in (workspace / '.webui', preview_root, directory, output_path, output_mask_path):
        if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
            raise ValueError('Preview output directories cannot be symbolic links')
    for candidate in (output_path, output_mask_path):
        if candidate.exists() and any(candidate.iterdir()):
            raise ValueError('Preview output directory must be unused; create a new preview task')
    return directory


def main (model_class_name=None,
          saved_models_path=None,
          training_data_src_path=None,
          force_model_name=None,
          input_path=None,
          output_path=None,
          output_mask_path=None,
          aligned_path=None,
          force_gpu_idxs=None,
          cpu_only=None,
          xseg_models_path=None,
          preview_frame_start=1,
          preview_frame_count=None):
    io.log_info ("正在运行合成器（Merger）。\r\n")

    rpc_functions = []
    model = None
    try:
        if not input_path.exists():
            raise FileNotFoundError(f'未找到输入目录：{input_path}')

        preview_directory = None
        input_path_image_paths = pathex.get_image_paths(input_path)
        total_source_frames = len(input_path_image_paths)
        source_frame_inventory_sha256 = hashlib.sha256('\n'.join(Path(item).name for item in input_path_image_paths).encode('utf-8')).hexdigest()
        if preview_frame_count is not None:
            preview_directory = validate_preview_output_paths(input_path, output_path, output_mask_path)
            input_path_image_paths = select_preview_frames(input_path_image_paths, preview_frame_start, preview_frame_count)
            if not os.environ.get('DFL_WEB_MERGE_CONFIG', '').strip():
                raise ValueError('Preview requires explicit guided merge parameters')
        elif preview_frame_start != 1:
            raise ValueError('Preview frame start requires preview frame count')

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not output_mask_path.exists():
            output_mask_path.mkdir(parents=True, exist_ok=True)

        if not saved_models_path.exists():
            raise FileNotFoundError(f'未找到模型目录：{saved_models_path}')

        # 初始化模型
        import models
        model = models.import_model(model_class_name)(is_training=False,
                                                      saved_models_path=saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      force_model_name=force_model_name,
                                                      cpu_only=cpu_only)

        predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()

        # Preparing MP functions
        predictor_func = MPFunc(predictor_func)
        rpc_functions.append(predictor_func)

        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
        xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',
                                                    name='XSeg',
                                                    resolution=256,
                                                    weights_file_root=xseg_models_path or saved_models_path,
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)
        rpc_functions.append(xseg_256_extract_func)


        web_config = os.environ.get('DFL_WEB_MERGE_CONFIG', '').strip()
        if web_config:
            subprocess_count = apply_web_merge_config(cfg, web_config)
            is_interactive = False
            io.log_info('已应用 Web 合成参数；不会打开 Merger 窗口。')
        else:
            is_interactive = io.input_bool ("是否使用交互式合成？", True) if not io.is_colab() else False
            if not is_interactive:
                cfg.ask_settings()
            subprocess_count = io.input_int(
                "工作进程数量？", min(8, multiprocessing.cpu_count()),
                valid_range=[1, multiprocessing.cpu_count()],
                help_message="指定用于处理的线程/进程数量；该值不得超过 CPU 核心数。",
            )

        from facelib.FaceEnhancement import AlignedFaceEnhancer, validate_enhancement_assets
        if cfg.super_resolution_power:
            validate_enhancement_assets(cfg.super_resolution_model)
        face_enhancer_func = MPClassFuncOnDemand(AlignedFaceEnhancer, 'enhance',
                                                model_id=cfg.super_resolution_model,
                                                device='cpu' if run_on_cpu else 'cuda')
        rpc_functions.append(face_enhancer_func)

        if cfg.type == MergerConfig.TYPE_MASKED:
            if not aligned_path.exists():
                raise FileNotFoundError(f'未找到 aligned 目录：{aligned_path}')

            packed_samples = samplelib.PackedFaceset.load(aligned_path)
            packed_identities = {}

            if packed_samples is not None:
                io.log_info ("检测到 PackedFaceset，将使用打包 faceset。")
                def generator():
                    for sample in io.progress_bar_generator(packed_samples, "正在收集对齐信息"):
                        filepath, dflimg, digest = read_packed_alignment(sample)
                        packed_identities[filepath.as_posix()] = digest
                        yield filepath, dflimg
            else:
                def generator():
                    for filepath in io.progress_bar_generator(pathex.get_image_paths(aligned_path), "正在收集对齐信息"):
                        filepath = Path(filepath)
                        yield filepath, DFLIMG.load(filepath)

            alignments = {}
            multiple_faces_detected = False

            for filepath, dflimg in generator():
                if dflimg is None or not dflimg.has_data():
                    io.log_err (f"{filepath.name} 不是 DFL 图像文件")
                    continue

                source_filename = dflimg.get_source_filename()
                if source_filename is None:
                    continue

                source_filepath = Path(source_filename)
                source_filename_stem = source_filepath.stem

                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments_ar = alignments[ source_filename_stem ]
                alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath ) )

                if len(alignments_ar) > 1:
                    multiple_faces_detected = True

            if multiple_faces_detected:
                io.log_info ("")
                io.log_info ("警告：检测到多张人脸。一个源文件通常应只对应一个 alignment 文件。")
                io.log_info ("")

            for a_key in list(alignments.keys()):
                a_ar = alignments[a_key]
                if len(a_ar) > 1:
                    for _, filepath, source_filepath in a_ar:
                        io.log_info (f"alignment {filepath.name} 指向源文件 {source_filepath.name}")
                    io.log_info ("")

                # Keep aligned identity and affine provenance alongside original source points.

            if multiple_faces_detected:
                io.log_info ("强烈建议将不同人脸分别处理（拆分 faceset）。")
                io.log_info ("可使用“恢复 aligned 原始文件名”来确定具体的重复项。")
                io.log_info ("")

            frames, merge_audit = prepare_merge_frames(input_path_image_paths, alignments, cfg, aligned_path,
                                                       packed_identities=packed_identities)
            if preview_directory is not None:
                merge_audit['preview'] = {'kind': 'bounded-independent-preview',
                    'frameStart': preview_frame_start, 'frameCount': preview_frame_count,
                    'totalSourceFrames': total_source_frames, 'modelName': model.name,
                    'sourceFrameInventorySha256': source_frame_inventory_sha256,
                    'modelSha256': sha256(model.directory / 'me.pt'),
                    'alignedPath': str(aligned_path.resolve().relative_to(input_path.resolve().parent)),
                    'requestedParameters': json.loads(web_config),
                    'device': {'cpuOnly': bool(cpu_only), 'gpuIndexes': force_gpu_idxs or ''},
                    'configurationSource': 'DFL_WEB_MERGE_CONFIG', 'notFullSequence': True}
            write_json(output_path / 'merge.audit.json', merge_audit)


        if len(frames) == 0:
            raise ValueError("输入目录中没有可合成的帧。")
        else:
            if False:
                pass
            else:
                InteractiveMergerSubprocessor (
                            is_interactive         = is_interactive,
                            merger_session_filepath = (preview_directory / 'merger_session.dat') if preview_directory is not None else model.get_strpath_storage_for_file('merger_session.dat'),
                            predictor_func         = predictor_func,
                            predictor_input_shape  = predictor_input_shape,
                            face_enhancer_func     = face_enhancer_func,
                            xseg_256_extract_func = xseg_256_extract_func,
                            merger_config          = cfg,
                            frames                 = frames,
                            frames_root_path       = input_path,
                            output_path            = output_path,
                            output_mask_path       = output_mask_path,
                            model_iter             = model.get_iter(),
                            subprocess_count       = subprocess_count,
                        ).run()

        merge_audit["status"] = "complete"
        for item, frame in zip(merge_audit["frames"], frames):
            result = output_path / (frame.frame_info.filepath.stem + ".png")
            mask_result = output_mask_path / result.name
            item["outputSha256"] = sha256(result) if result.is_file() else None
            item["outputMaskSha256"] = sha256(mask_result) if mask_result.is_file() else None
        if any(item["outputSha256"] is None or item["outputMaskSha256"] is None for item in merge_audit["frames"]):
            merge_audit["status"] = "incomplete"
        write_json(output_path / "merge.audit.json", merge_audit)
        if preview_directory is not None and merge_audit['status'] != 'complete':
            raise ValueError('Preview output sequence is incomplete')

    except Exception as e:
        if 'merge_audit' in locals():
            merge_audit['status'] = 'failed'
            merge_audit['error'] = str(e)
            write_json(output_path / 'merge.audit.json', merge_audit)
        print ( traceback.format_exc() )
        raise
    finally:
        cleanup_errors = []
        for function in reversed(rpc_functions):
            try:
                close = getattr(function, 'close', None)
                if close is not None:
                    close()
            except Exception as error:
                cleanup_errors.append(error)
        if model is not None:
            try:
                model.finalize()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise ExceptionGroup('Merger helper shutdown could not be confirmed', cleanup_errors)




#插值 landmarks（关键点）
#from facelib import LandmarksProcessor
#from facelib import FaceType
#a = sorted(alignments.keys())
#a_len = len(a)
#
#box_pts = 3
#box = np.ones(box_pts)/box_pts
#for i in range( a_len ):
#    if i >= box_pts and i <= a_len-box_pts-1:
#        af0 = alignments[ a[i] ][0] ##first face
#        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)
#
#        points = []
#
#        for j in range(-box_pts, box_pts+1):
#            af = alignments[ a[i+j] ][0] ##first face
#            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)
#            p = LandmarksProcessor.transform_points (af, m)
#            points.append (p)
#
#        points = np.array(points)
#        points_len = len(points)
#        t_points = np.transpose(points, [1,0,2])
#
#        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
#        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
#
#        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
#
#        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
