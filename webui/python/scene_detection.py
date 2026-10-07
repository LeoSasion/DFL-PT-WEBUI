"""Read-only scene candidates with original frame indices and integer source PTS.

FFmpeg is the default. Optional algorithms never download models or install
packages. Frame-count algorithms receive decoded frames in source order; their
indices are mapped back to the independently probed source timeline.
"""
import argparse
import contextlib
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))
from core.media_timeline import file_sha256, probe_timeline


def progress(message):
    print(message, file=sys.stderr, flush=True)


def terminal_pts(timeline):
    from core.media_timeline import verified_terminal_pts
    return verified_terminal_pts(timeline)


def assemble(timeline, detected, algorithm, parameters, limit=500):
    frames = timeline['frames']
    end_pts, tail_policy = terminal_pts(timeline)
    by_index = {}
    for item in detected:
        index = item['sourceFrameIndex']
        if not isinstance(index, int) or not 0 <= index < len(frames):
            raise ValueError('Scene candidate reported an invalid original frame index')
        if index == 0:
            continue
        cut = {key: frames[index][key] for key in ('sourceFrameIndex', 'pts', 'timeBase')}
        if item.get('score') is not None:
            cut['score'] = float(item['score'])
            if not math.isfinite(cut['score']):
                raise ValueError('Scene candidate reported a nonfinite score')
        by_index[index] = cut
    all_cuts = [by_index[index] for index in sorted(by_index)]
    total = len(all_cuts) + 1
    # Keep the first reviewable cuts and merge the unlisted remainder into one
    # final scene. The last source frame is always represented.
    cuts = all_cuts[:limit - 1]
    boundaries = [0, *[cut['sourceFrameIndex'] for cut in cuts], len(frames)]
    tb = Fraction(*frames[0]['timeBase'])
    origin = frames[0]['pts']
    scenes = []
    for start_index, end_index in zip(boundaries, boundaries[1:]):
        start = frames[start_index]['pts']
        end = frames[end_index]['pts'] if end_index < len(frames) else end_pts
        scenes.append({'start': float((start - origin) * tb), 'end': float((end - origin) * tb),
                       'startFrameIndex': start_index, 'endFrameIndexExclusive': end_index,
                       'startPts': start, 'endPts': end, 'timeBase': frames[0]['timeBase']})
    return {'schemaVersion': 2, 'kind': 'scene-detection', 'source': timeline['source'],
            'algorithm': algorithm, 'parameters': parameters,
            'sourceFrameCount': len(frames), 'firstPts': origin, 'terminalPts': end_pts,
            'timeBase': frames[0]['timeBase'], 'terminalDurationPolicy': tail_policy,
            'cuts': all_cuts, 'scenes': scenes, 'total': total, 'truncated': total > limit,
            'sceneLimit': limit, 'truncationPolicy': 'first-499-cuts-plus-final-tail',
            'modelProvenance': parameters.get('modelProvenance'),
            'colorContract': timeline.get('colorContract')}


def ffmpeg_cuts(source, ffmpeg, timeline, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('FFmpeg threshold must be between 0 and 1')
    by_pts = {frame['pts']: frame['sourceFrameIndex'] for frame in timeline['frames']}
    cuts, current = [], None
    with tempfile.TemporaryFile() as log:
        result = subprocess.run([str(ffmpeg), '-hide_banner', '-nostdin', '-copyts', '-i', str(source),
            '-map', '0:v:0', '-an', '-vf', f"select='gt(scene,{threshold:.12g})',metadata=mode=print",
            '-fps_mode', 'passthrough', '-f', 'null', '-'], stdout=subprocess.DEVNULL, stderr=log)
        log.seek(0)
        recent = []
        for raw in log:
            line = raw.decode('utf-8', 'replace')
            recent = (recent + [line])[-10:]
            match = re.search(r'\bframe:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:', line)
            if match:
                current = int(match.group(1))
            score = re.search(r'lavfi\.scene_score=([0-9.eE+-]+)', line)
            if score:
                if current not in by_pts:
                    raise RuntimeError('FFmpeg selected PTS is absent from the source timeline')
                cuts.append({'sourceFrameIndex': by_pts[current], 'score': float(score.group(1))})
        if result.returncode:
            raise RuntimeError('FFmpeg scene detection failed: ' + ''.join(recent))
    return cuts, {'threshold': threshold, 'score': 'ffmpeg-lavfi.scene_score', 'frameMapping': 'original-integer-pts'}


@contextlib.contextmanager
def decoded_rgb(source, ffmpeg, width, height, count):
    import numpy as np
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([str(ffmpeg), '-v', 'error', '-nostdin', '-i', str(source),
            '-map', '0:v:0', '-an', '-vf', f'scale={width}:{height}:flags=bilinear',
            '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'],
            stdout=subprocess.PIPE, stderr=log)
        def stream():
            for index in range(count):
                chunks, remaining = [], width * height * 3
                while remaining:
                    chunk = process.stdout.read(remaining)
                    if not chunk:
                        raise RuntimeError('Decoded frame count is shorter than the original source timeline')
                    chunks.append(chunk)
                    remaining -= len(chunk)
                yield index, np.frombuffer(b''.join(chunks), np.uint8).reshape(height, width, 3)
            if process.stdout.read(1):
                raise RuntimeError('Decoded frame count exceeds the original source timeline')
            if process.wait() != 0:
                log.seek(0)
                raise RuntimeError('FFmpeg decoding failed: ' + log.read().decode('utf-8', 'replace')[-4000:])
        try:
            yield stream()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def adaptive_cuts(source, ffmpeg, timeline, threshold):
    if threshold <= 0 or threshold > 255:
        raise ValueError('Adaptive ratio threshold must be above 0 and at most 255')
    from scenedetect import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector
    detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=15, window_width=2, min_content_val=15)
    cuts = []
    with decoded_rgb(source, ffmpeg, 320, 180, len(timeline['frames'])) as frames:
        for index, rgb in frames:
            # Only frame-count comparisons use this index timecode. The public
            # contract never derives source timing from an assumed constant FPS.
            for cut in detector.process_frame(FrameTimecode(index, Fraction(1)), rgb[:, :, ::-1].copy()):
                cuts.append({'sourceFrameIndex': cut.get_frames()})
            if index and index % 250 == 0:
                progress(f'adaptive {index}/{len(timeline["frames"])} source frames')
    return cuts, {'threshold': threshold, 'minSceneFrames': 15, 'windowWidth': 2, 'minContentValue': 15,
                  'decodeSize': [320, 180], 'implementation': 'PySceneDetect AdaptiveDetector 0.7.1',
                  'frameMapping': 'decoded-source-frame-index-to-original-pts'}


def transnet_cuts(source, ffmpeg, timeline, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('TransNet threshold must be between 0 and 1')
    import numpy as np
    import torch
    resources = verified_scene_resources(ROOT)
    identity = json.loads((resources / 'identity.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('_dfl_official_transnetv2', resources / 'transnetv2_pytorch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransNetV2().eval()
    model.load_state_dict(torch.load(resources / 'transnetv2-pytorch-weights.pth', map_location='cpu', weights_only=True), strict=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    # Official overlap convention: 25 frames of context at either side of a
    # 50-frame evaluated core, repeating terminal frames only for context.
    scores = np.empty(len(timeline['frames']), np.float32)
    with decoded_rgb(source, ffmpeg, 48, 27, len(scores)) as stream:
        iterator = iter(stream)
        buffered = []
        first = next(iterator)[1]
        buffered.extend([first] * 25)
        buffered.append(first)
        exhausted = False
        offset = 0
        with torch.inference_mode():
            while offset < len(scores):
                while len(buffered) < 100:
                    if exhausted:
                        buffered.append(buffered[-1])
                    else:
                        try:
                            buffered.append(next(iterator)[1])
                        except StopIteration:
                            exhausted = True
                batch = torch.from_numpy(np.stack(buffered[:100])[None])
                single, _ = model(batch)
                prediction = torch.sigmoid(single).numpy().reshape(-1)[25:75]
                size = min(50, len(scores) - offset)
                scores[offset:offset + size] = prediction[:size]
                offset += size
                del buffered[:50]
                progress(f'transnetv2 {offset}/{len(scores)} source frames')
            # Exhaust the decoder generator to validate frame count/exit status.
            for _ in iterator:
                pass
    # Official predictions_to_scenes starts a new scene at the 1->0 edge;
    # active prediction frames describe the preceding transition/end of shot.
    # Retain transition frames in the previous segment rather than dropping
    # them, and map the first following source frame to the new segment.
    active = False
    runs, run = [], None
    for index, score in enumerate(scores):
        selected = float(score) > threshold
        if selected and not active:
            run = {'startFrameIndex': index, 'endFrameIndexExclusive': index + 1, 'maxScore': float(score)}
        elif selected:
            run['endFrameIndexExclusive'] = index + 1
            run['maxScore'] = max(run['maxScore'], float(score))
        elif active:
            runs.append(run)
        active = selected
    if active:
        runs.append(run)
    cuts = [{'sourceFrameIndex': run['endFrameIndexExclusive'], 'score': run['maxScore']}
            for run in runs if run['endFrameIndexExclusive'] < len(scores)]
    return cuts, {'threshold': threshold, 'decodeSize': [48, 27], 'contextFrames': 25, 'coreFrames': 50,
                  'runtime': 'official-PyTorch-CPU-FP32', 'transitionPolicy': 'first-frame-after-threshold-run-official-1-to-0-edge',
                  'scoreSource': 'transition-run-maximum',
                  'transitionRuns': runs, 'frameMapping': 'decoded-source-frame-index-to-original-pts',
                  'modelProvenance': {'repository': identity['repository'], 'revision': identity['revision'],
                    'identitySha256': file_sha256(resources / 'identity.json'),
                    'weightSha256': file_sha256(resources / 'transnetv2-pytorch-weights.pth'),
                    'everyTensorInversePermutationExact': identity['everyTensorInversePermutationExact'],
                    'tensorflowRuntimeImported': False, 'tensorflowForwardEquivalenceTested': False}}


def verified_scene_resources(root):
    manifest = json.loads((root / 'release/vision-assets.json').read_text(encoding='utf-8'))
    group = next((item for item in manifest['groups'] if item['id'] == 'transnetv2'), None)
    if group is None:
        raise ValueError('TransNetV2 has no admitted fixed project resource group')
    directory = root / group['installRoot']
    for item in group['files']:
        path = directory / item['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item['sizeBytes'] or file_sha256(path) != item['sha256']:
            raise ValueError('TransNetV2 resource missing/changed; explicitly install the scene resource pack: ' + item['path'])
    return directory


def capabilities():
    algorithms = [{'id': 'ffmpeg', 'available': True}]
    try:
        from importlib.metadata import version
        installed = version('scenedetect')
        if installed != '0.7.1':
            raise ValueError('Expected fixed PySceneDetect 0.7.1; installed ' + installed)
        from scenedetect.detectors import AdaptiveDetector
        algorithms.append({'id': 'adaptive', 'available': True})
    except Exception as error:
        algorithms.append({'id': 'adaptive', 'available': False, 'reason': str(error)})
    try:
        import torch
        resources = verified_scene_resources(ROOT)
        algorithms.append({'id': 'transnetv2', 'available': True,
                           'identitySha256': file_sha256(resources / 'identity.json')})
    except Exception as error:
        algorithms.append({'id': 'transnetv2', 'available': False, 'reason': str(error)})
    return {'schemaVersion': 2, 'algorithms': algorithms}


def detect(source, ffmpeg, ffprobe, algorithm='ffmpeg', threshold=None):
    source = Path(source).resolve(strict=True)
    initial_size, initial_sha = source.stat().st_size, file_sha256(source)
    timeline = probe_timeline(source, ffprobe)
    if initial_size != timeline['source']['bytes'] or initial_sha != timeline['source']['sha256']:
        raise RuntimeError('Source changed during timeline probing; results are not published')
    progress(f'{algorithm}: {len(timeline["frames"])} original source frames')
    threshold = {'ffmpeg': 0.32, 'adaptive': 3.0, 'transnetv2': 0.5}[algorithm] if threshold is None else threshold
    if not math.isfinite(threshold):
        raise ValueError('Scene threshold must be finite')
    function = {'ffmpeg': ffmpeg_cuts, 'adaptive': adaptive_cuts, 'transnetv2': transnet_cuts}[algorithm]
    # Third-party initialization output belongs to diagnostics, never JSON stdout.
    with contextlib.redirect_stdout(sys.stderr):
        cuts, parameters = function(source, ffmpeg, timeline, threshold)
    if source.stat().st_size != timeline['source']['bytes'] or file_sha256(source) != timeline['source']['sha256']:
        raise RuntimeError('Source changed during scene detection; results are not published')
    return assemble(timeline, cuts, algorithm, parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--ffprobe', type=Path)
    parser.add_argument('--capabilities', action='store_true')
    parser.add_argument('--algorithm', choices=('ffmpeg', 'adaptive', 'transnetv2'), default='ffmpeg')
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    try:
        if args.capabilities:
            with contextlib.redirect_stdout(sys.stderr):
                result = capabilities()
            print(json.dumps(result))
            raise SystemExit(0)
        if not all((args.source, args.ffmpeg, args.ffprobe)):
            parser.error('--source, --ffmpeg and --ffprobe are required for detection')
        result = detect(args.source, args.ffmpeg, args.ffprobe, args.algorithm, args.threshold)
        print(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    except Exception as error:
        progress(f'{type(error).__name__}: {error}')
        print(json.dumps({'schemaVersion': 2, 'ok': False, 'algorithm': args.algorithm,
                          'error': str(error), 'errorType': type(error).__name__}, ensure_ascii=False))
        raise SystemExit(2)
