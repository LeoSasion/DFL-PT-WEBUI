import subprocess
import shutil
import tempfile
import numpy as np
import ffmpeg
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

    for filename in pathex.get_image_paths(output_path, ['.' + output_ext]):
        Path(filename).unlink()

    job = ffmpeg.input(str(input_file_path))

    kwargs = {'pix_fmt': 'rgb24'}
    if fps != 0:
        kwargs.update({'r': str(fps)})

    if output_ext == 'jpg':
        kwargs.update({'q:v': '2'})  # highest quality for jpg

    job = job.output(str(output_path / ('%5d.' + output_ext)), **kwargs)

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')

    try:
        job = job.run(cmd=ffmpeg_cmd)
    except ffmpeg.Error as error:
        io.log_err("ffmpeg 执行失败，命令行：" + str(job.compile(cmd=ffmpeg_cmd)))
        raise RuntimeError("FFmpeg processing failed") from error


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

    kwargs = {
        "c:v": "libx264",
        "b:v": "%dM" % (bitrate),
        "pix_fmt": "yuv420p",
    }

    job = ffmpeg.input(str(input_file_path), ss=from_time, to=to_time)

    job_v = job['v:0']
    job_a = job['a:' + str(audio_track_id) + '?']

    job = ffmpeg.output(job_v, job_a, str(output_file_path), **kwargs).overwrite_output()

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')

    try:
        job = job.run(cmd=ffmpeg_cmd)
    except ffmpeg.Error as error:
        io.log_err("ffmpeg 执行失败，命令行：" + str(job.compile(cmd=ffmpeg_cmd)))
        raise RuntimeError("FFmpeg processing failed") from error


def denoise_image_sequence(input_dir, ext=None, factor=None):
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

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')
    # Separate FFmpeg's input and output and validate every frame before replacing
    # originals. The old in-place sequence could truncate its own input files.
    from core.cv2ex import cv2_imread
    from DFLIMG import DFLIMG
    with tempfile.TemporaryDirectory(prefix='.denoise-', dir=input_path.parent) as temporary:
        staging = Path(temporary)
        source, result, backup = [staging / name for name in ('input', 'output', 'original')]
        for folder in (source, result, backup):
            folder.mkdir()
        for index, path in enumerate(image_paths, 1):
            shutil.copyfile(path, source / f'{index:06d}{image_paths_suffix}')
        kwargs = {'q:v': '2'} if image_paths_suffix in ('.jpg', '.jpeg') else {}
        pattern = '%06d' + image_paths_suffix
        job = (ffmpeg.input(str(source / pattern), start_number=1)
               .filter('hqdn3d', factor, factor, 5, 5)
               .output(str(result / pattern), start_number=1, **kwargs).overwrite_output())
        try:
            job.run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as error:
            raise RuntimeError('FFmpeg denoising failed') from error
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
        replaced = []
        try:
            for original, output in zip(image_paths, processed):
                saved = backup / original.name
                original.replace(saved)
                replaced.append((original, saved))
                output.replace(original)
        except Exception:
            for original, saved in reversed(replaced):
                saved.replace(original)
            raise


def video_from_sequence(input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, include_audio=False, lossless=None):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    reference_file_path = Path(reference_file) if reference_file is not None else None

    if not input_path.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{input_path}")

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    out_ext = output_file_path.suffix

    if ext is None:
        ext = io.input_str("输入图片格式（扩展名）", "png")

    if lossless is None:
        lossless = io.input_bool("是否使用无损编码", False)

    video_id = None
    audio_id = None
    ref_in_a = None
    if reference_file_path is not None:
        if reference_file_path.suffix == '.*':
            reference_file_path = pathex.get_first_file_by_stem(reference_file_path.parent, reference_file_path.stem)
        else:
            if not reference_file_path.exists():
                reference_file_path = None

        if reference_file_path is None:
            raise FileNotFoundError("未找到参考视频。")

        # probing reference file
        ffprobe_cmd = osex.get_ffprobe_path()
        if ffprobe_cmd is None:
            raise FileNotFoundError('未找到 ffprobe。请安装 ffmpeg（通常会自带 ffprobe），或将 ffprobe 可执行文件放到 tools/ 目录（Windows：tools/ffprobe.exe），或设置环境变量 DFL_FFPROBE 指向 ffprobe。')

        probe = ffmpeg.probe(str(reference_file_path), cmd=ffprobe_cmd)

        # getting first video and audio streams id with fps
        for stream in probe['streams']:
            if video_id is None and stream['codec_type'] == 'video':
                video_id = stream['index']
                fps = stream['r_frame_rate']

            if audio_id is None and stream['codec_type'] == 'audio':
                audio_id = stream['index']

        if audio_id is not None:
            # has audio track
            ref_in_a = ffmpeg.input(str(reference_file_path))[str(audio_id)]

    if fps is None:
        # if fps not specified and not overwritten by reference-file
        fps = max(1, io.input_int("请输入 FPS", 25))

    if not lossless and bitrate is None:
        bitrate = max(1, io.input_int("输出视频码率（MB/s）", 16))

    input_image_paths = pathex.get_image_paths(input_path, image_extensions=['.' + ext.lstrip('.')])
    if not input_image_paths:
        raise ValueError("输入目录没有可封装的图片。")

    i_in = ffmpeg.input('pipe:', format='image2pipe', r=fps)

    output_args = [i_in]

    if include_audio and ref_in_a is not None:
        output_args += [ref_in_a]

    output_args += [str(output_file_path)]

    output_kwargs = {}

    if lossless:
        output_kwargs.update({
            "c:v": "libx264",
            "crf": "0",
            "pix_fmt": "yuv420p",
        })
    else:
        output_kwargs.update({
            "c:v": "libx264",
            "b:v": "%dM" % (bitrate),
            "pix_fmt": "yuv420p",
        })

    ffmpeg_cmd = osex.get_ffmpeg_path()
    if ffmpeg_cmd is None:
        raise FileNotFoundError('未找到 ffmpeg。请安装 ffmpeg，或将 ffmpeg 可执行文件放到 tools/ 目录（Windows：tools/ffmpeg.exe），或设置环境变量 DFL_FFMPEG 指向 ffmpeg。')

    if include_audio and ref_in_a is not None:
        output_kwargs.update({
            "c:a": "aac",
            "b:a": "192k",
            "ar": "48000",
            "strict": "experimental",
        })

    job = (ffmpeg.output(*output_args, **output_kwargs).overwrite_output())

    try:
        job_run = job.run_async(cmd=ffmpeg_cmd, pipe_stdin=True)

        for image_path in input_image_paths:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                job_run.stdin.write(image_bytes)

        job_run.stdin.close()
        return_code = job_run.wait()
        if return_code != 0 or not output_file_path.is_file() or output_file_path.stat().st_size == 0:
            raise RuntimeError(f'FFmpeg video export failed (exit {return_code})')
    except (OSError, ffmpeg.Error) as error:
        io.log_err("ffmpeg 失败，任务命令行：" + str(job.compile(cmd=ffmpeg_cmd)))
        raise RuntimeError("FFmpeg video export failed") from error
