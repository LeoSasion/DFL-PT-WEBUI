import subprocess
import shutil
import tempfile
import numpy as np
from pathlib import Path
from core import pathex
from core import osex
from core.interact import interact as io


def extract_video(input_file, output_dir, output_ext=None, fps=None):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if input_file_path.suffix == '.*':
        input_file_path = pathex.get_first_file_by_stem(input_file_path.parent, input_file_path.stem)
    else:
        if not input_file_path.exists():
            input_file_path = None

    if input_file_path is None:
        raise FileNotFoundError("未找到输入视频。")

    if fps is None:
        fps = io.input_int("输入 FPS", 0, help_message="每秒提取多少帧。0 表示使用视频原始全帧率")

    if output_ext is None:
        output_ext = io.input_str(
            "输出图片格式",
            "png",
            ["png", "jpg"],
            help_message="png 为无损，但在机械硬盘上提取速度可能更慢，并且比 jpg 占用更多磁盘空间。",
        )

    from core.media_timeline import extract_frames, MANIFEST_NAME
    ffmpeg_cmd, ffprobe_cmd = osex.get_ffmpeg_path(), osex.get_ffprobe_path()
    if not ffmpeg_cmd or not ffprobe_cmd:
        raise FileNotFoundError('提帧需要项目内 FFmpeg 与 ffprobe。')
    # Keep the prior image sequence and its timeline together, and publish only
    # after decoded source PTS and every extracted file have been checked.
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=output_path.parent) as temporary:
        staging = Path(temporary) / 'new'
        timeline = extract_frames(input_file_path, staging, ffmpeg_cmd, ffprobe_cmd, output_ext, fps)
        import uuid
        backup = output_path / ('.frame-extract-backup-' + uuid.uuid4().hex)
        existing = [Path(item) for item in pathex.get_image_paths(output_path)]
        if (output_path / MANIFEST_NAME).exists():
            existing.append(output_path / MANIFEST_NAME)
        moved, installed = [], []
        try:
            if existing:
                backup.mkdir()
            for image in existing:
                saved = backup / image.name
                image.replace(saved)
                moved.append((image, saved))
            for result in staging.iterdir():
                destination = output_path / result.name
                result.replace(destination)
                installed.append(destination)
        except Exception:
            for result in installed:
                if result.exists():
                    result.replace(staging / result.name)
            for original, saved in reversed(moved):
                saved.replace(original)
            raise
        io.log_info(f'已提取 {len(timeline["frames"])} 帧，源 PTS/颜色/音轨清单：{MANIFEST_NAME}')
        return timeline


def cut_video(input_file, from_time=None, to_time=None, audio_track_id=None, bitrate=None):
    input_file_path = Path(input_file)
    if not input_file_path.is_file():
        raise FileNotFoundError(f"未找到输入视频：{input_file_path}")

    output_file_path = input_file_path.parent / (input_file_path.stem + "_cut" + input_file_path.suffix)

    if from_time is None:
        from_time = io.input_str("起始时间", "00:00:00.000")

    if to_time is None:
        to_time = io.input_str("结束时间", "00:00:00.000")

    if audio_track_id is None:
        audio_track_id = io.input_int("指定音轨 ID", 0)

    if bitrate is None:
        bitrate = max(1, io.input_int("输出码率（Mbps）", 25))

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')

    from core.media_timeline import run
    if isinstance(audio_track_id, bool) or int(audio_track_id) != audio_track_id or int(audio_track_id) < 0:
        raise ValueError('Audio track index must be a nonnegative integer')
    if not 1 <= int(bitrate) <= 1000:
        raise ValueError('Video bitrate must be between 1 and 1000 Mbps')
    # Each value is one fixed argument, including Unicode/space-containing paths.
    # A failed encoding cannot overwrite a previous successful cut.
    import uuid
    pending = output_file_path.with_name('.cut-' + uuid.uuid4().hex + output_file_path.suffix)
    try:
        run([ffmpeg_cmd, '-hide_banner', '-nostdin', '-y', '-ss', str(from_time), '-to', str(to_time),
             '-i', input_file_path, '-map', '0:v:0', '-map', f'0:a:{int(audio_track_id)}?',
             '-c:v', 'libx264', '-b:v', f'{int(bitrate)}M', '-pix_fmt', 'yuv420p', pending])
        if not pending.is_file() or not pending.stat().st_size:
            raise RuntimeError('FFmpeg produced an empty cut')
        if output_file_path.exists():
            backup = output_file_path.with_name(output_file_path.stem + '.backup-' + uuid.uuid4().hex + output_file_path.suffix)
            shutil.copyfile(output_file_path, backup)
        pending.replace(output_file_path)
    finally:
        pending.unlink(missing_ok=True)


def denoise_image_sequence(input_dir, ext=None, factor=None):
    import json
    from core.faceset_transaction import execute_plan, sha256
    from core.media_timeline import MANIFEST_NAME
    from merger.temporal_geometry import bind_timeline

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{input_path}")

    extensions = ['.' + ext.lstrip('.').lower()] if ext else None
    image_paths = [Path(filepath) for filepath in pathex.get_image_paths(input_path, **({'image_extensions': extensions} if extensions else {}))]
    if not image_paths:
        raise ValueError("输入目录没有可降噪的图片。")

    # Check extension of all images
    image_paths_suffix = None
    for filepath in image_paths:
        if image_paths_suffix is None:
            image_paths_suffix = filepath.suffix
        else:
            if filepath.suffix != image_paths_suffix:
                raise ValueError(f"{input_path.name} 目录内所有图片必须使用相同的扩展名。")

    if factor is None:
        factor = np.clip(io.input_int("降噪强度？", 7, add_info="1-20"), 1, 20)

    factor = float(factor)
    if not np.isfinite(factor) or not 1 <= factor <= 20:
        raise ValueError('Denoising factor must be between 1 and 20')

    # A denoised frame retains its original PTS, but has different pixels.
    # Validate the current binding before processing; never bless an already
    # stale manifest by merely replacing its hashes after the fact.
    timeline_path = input_path / MANIFEST_NAME
    timeline = None
    if timeline_path.exists():
        bind_timeline(image_paths)
        timeline = json.loads(timeline_path.read_text(encoding='utf-8'))
        if not isinstance(timeline.get('processingHistory', []), list):
            raise ValueError('Invalid source-frame processing history')
    source_hashes = {path.name: sha256(path) for path in image_paths}
    timeline_hash = sha256(timeline_path) if timeline is not None else None

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
    # Separate FFmpeg's input and output, then publish pictures and their timing
    # manifest through one recoverable batch with durable original backups.
    from core.cv2ex import cv2_imread
    from DFLIMG import DFLIMG
    with tempfile.TemporaryDirectory(prefix='.denoise-', dir=input_path.parent) as temporary:
        staging = Path(temporary)
        source, result = [staging / name for name in ('input', 'output')]
        for folder in (source, result):
            folder.mkdir()
        for index, path in enumerate(image_paths, 1):
            shutil.copyfile(path, source / f'{index:06d}{image_paths_suffix}')
        pattern = '%06d' + image_paths_suffix
        from core.media_timeline import run
        quality = ['-q:v', '2'] if image_paths_suffix in ('.jpg', '.jpeg') else []
        run([ffmpeg_cmd, '-hide_banner', '-nostdin', '-y', '-start_number', '1', '-i', source / pattern,
             '-vf', f'hqdn3d={factor}:{factor}:5:5', '-start_number', '1', *quality, result / pattern])
        processed = sorted(result.glob('*' + image_paths_suffix))
        if len(processed) != len(image_paths):
            raise RuntimeError('FFmpeg denoising returned an incomplete image sequence')
        for original, output in zip(image_paths, processed):
            original_pixels, output_pixels = cv2_imread(original), cv2_imread(output)
            if output_pixels is None or original_pixels is None or output_pixels.shape != original_pixels.shape:
                raise RuntimeError(f'Invalid denoised frame: {original.name}')
            metadata = DFLIMG.load(original)
            if metadata is not None and metadata.has_data():
                denoised = DFLIMG.load(output)
                denoised.set_dict(metadata.get_dict())
                denoised.save()
        changes, records = [], []
        for original, output in zip(image_paths, processed):
            output_hash = sha256(output)
            changes.append({'source': original.name, 'target': original.name,
                            'source_sha256': source_hashes[original.name],
                            'payload_path': output, 'payload_sha256': output_hash})
            records.append({'file': original.name, 'inputImageSha256': source_hashes[original.name],
                            'outputImageSha256': output_hash})
        if timeline is not None:
            by_name = {frame['file']: frame for frame in timeline['frames']}
            for record in records:
                frame = by_name[record['file']]
                frame.setdefault('originalImageSha256', frame['imageSha256'])
                frame['imageSha256'] = record['outputImageSha256']
            timeline.setdefault('processingHistory', []).append({
                'operation': 'denoise', 'filter': 'hqdn3d',
                'parameters': {'lumaSpatial': factor, 'chromaSpatial': factor,
                               'lumaTemporal': 5, 'chromaTemporal': 5},
                'sourcePtsPreserved': True, 'frames': records})
            changes.append({'source': MANIFEST_NAME, 'target': MANIFEST_NAME,
                            'source_sha256': timeline_hash,
                            'payload': (json.dumps(timeline, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + '\n').encode('utf-8')})
        if (any(sha256(path) != source_hashes[path.name] for path in image_paths)
                or timeline is not None and sha256(timeline_path) != timeline_hash):
            raise RuntimeError('Source frames or timing manifest changed during denoising')
        receipt = execute_plan(input_path, changes, operation='denoise', details={
            'filter': 'hqdn3d', 'factor': factor, 'timingManifestUpdated': timeline is not None,
            'sourcePtsPreserved': True, 'frames': records})
        io.log_info('去噪图片与时间清单已一起提交；原件和恢复回执：' + receipt['receipt_path'])
        return receipt


def video_from_sequence(input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, include_audio=False, lossless=None):
    from core.media_timeline import encode_sequence
    input_path, target = Path(input_dir), Path(output_file)
    if not input_path.is_dir():
        raise FileNotFoundError(f'未找到输入目录：{input_path}')
    target.parent.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_file) if reference_file else None
    if reference and reference.suffix == '.*':
        reference = pathex.get_first_file_by_stem(reference.parent, reference.stem)
    if reference_file and (reference is None or not Path(reference).is_file()):
        raise FileNotFoundError('未找到参考视频。')
    ext = ext or io.input_str('输入图片格式（扩展名）', 'png')
    if lossless is None:
        lossless = io.input_bool('是否使用编码器无损模式（MP4仍有YUV颜色转换；RGB母版请输出.nut）', False)
    images = [Path(item) for item in pathex.get_image_paths(input_path, image_extensions=['.' + ext.lstrip('.')])]
    if not images:
        raise ValueError('输入目录没有可封装的图片。')
    if fps is None and reference is None and not any((parent / 'frames.timeline.json').is_file() for parent in (input_path, input_path.parent)):
        fps = max(1, io.input_int('请输入明确的恒定 FPS（没有源时间清单）', 25))
    if not lossless and bitrate is None:
        bitrate = max(1, io.input_int('输出视频码率（Mbps）', 16))
    ffmpeg_cmd, ffprobe_cmd = osex.get_ffmpeg_path(), osex.get_ffprobe_path()
    if not ffmpeg_cmd or not ffprobe_cmd:
        raise FileNotFoundError('视频封装需要项目内 FFmpeg 与 ffprobe。')
    record = encode_sequence(images, target, ffmpeg_cmd, ffprobe_cmd, reference=reference, fps=fps,
                             bitrate=bitrate or 16, lossless=lossless, include_audio=include_audio)
    io.log_info('视频导出已回读验证：' + record['mode'] + '；时间与验证记录：' + target.name + '.media.json')
    return record
