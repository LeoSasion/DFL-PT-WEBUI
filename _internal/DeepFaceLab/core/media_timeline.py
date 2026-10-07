"""Independent FFmpeg media contract: rational source timing and RGB round trips.

No source-video colour accuracy is inferred from successful decoding. The
working representation is FFmpeg-decoded RGB8, not a claim of calibrated sRGB.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

MANIFEST_NAME = 'frames.timeline.json'
COLOR_FIELDS = ('pix_fmt', 'color_range', 'color_space', 'color_transfer', 'color_primaries', 'chroma_location')


def file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    result = subprocess.run([str(item) for item in args], capture_output=True, timeout=3600)
    if result.returncode:
        raise RuntimeError('FFmpeg media operation failed: ' + result.stderr.decode('utf-8', 'replace')[-4000:])
    return result.stdout


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def probe_timeline(source, ffprobe):
    data = json.loads(run([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', source]))
    video = next((stream for stream in data['streams'] if stream['codec_type'] == 'video'), None)
    if video is None:
        raise ValueError('Source contains no video stream')
    tb = Fraction(video['time_base'])
    if tb <= 0:
        raise ValueError('Source has no valid video time base')
    decoded = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_entries',
                              'frame=pts,best_effort_timestamp,duration,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries',
                              '-of', 'json', source])).get('frames', [])
    frames, previous = [], None
    for index, frame in enumerate(decoded):
        if frame.get('color_transfer') in ('smpte2084', 'arib-std-b67') or frame.get('color_primaries') == 'bt2020':
            raise ValueError('HDR/wide-gamut frame colours require an explicit transform; RGB8 extraction is blocked')
        if 'pts' not in frame:
            raise ValueError('Decoded source frame lacks an original integer PTS; guessed timestamps are not accepted')
        pts = int(frame['pts'])
        if previous is not None and pts <= previous:
            raise ValueError('Decoded source PTS must increase strictly')
        previous = pts
        frames.append({'sourceFrameIndex': index, 'pts': pts, 'timeBase': [tb.numerator, tb.denominator],
                       'durationPts': int(frame['duration']) if int(frame.get('duration', 0)) > 0 else None,
                       'width': frame.get('width'), 'height': frame.get('height'),
                       'color': {field: frame.get(field, video.get(field, 'unknown')) for field in COLOR_FIELDS}})
    if not frames:
        raise ValueError('Source contains no decoded frames')
    known = all(video.get(key) not in (None, 'unknown', 'unspecified') for key in ('color_space', 'color_transfer', 'color_primaries', 'color_range'))
    hdr = video.get('color_transfer') in ('smpte2084', 'arib-std-b67') or video.get('color_primaries') == 'bt2020'
    if hdr:
        raise ValueError('HDR/wide-gamut sources require an explicit implemented colour transform; RGB8 extraction is blocked')
    first_audio_pts = {}
    if any(stream['codec_type'] == 'audio' for stream in data['streams']):
        packets = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'a', '-show_packets', '-read_intervals', '%+#128',
                                  '-show_entries', 'packet=stream_index,pts', '-of', 'json', source])).get('packets', [])
        for packet in packets:
            first_audio_pts.setdefault(packet['stream_index'], packet.get('pts'))
    return {'schemaVersion': 1, 'product': 'DFL-PT-WEBUI', 'kind': 'decoded-video-timeline',
            'source': {'name': Path(source).name, 'bytes': Path(source).stat().st_size, 'sha256': file_sha256(source)},
            'video': {key: video.get(key) for key in ('index', 'codec_name', 'time_base', 'start_pts', 'duration_ts', 'width', 'height', 'sample_aspect_ratio', *COLOR_FIELDS)},
            'colorContract': {'workingFormat': 'rgb24', 'conversion': 'ffmpeg-swscale-source-tags-to-rgb8',
                              'sourceTagsComplete': known, 'sourceColorAccuracyVerified': False,
                              'transferConvertedToSrgb': False, 'unknownTagsPolicy': 'recorded-ffmpeg-defaults-not-color-accuracy'},
            'audio': [{**{key: stream.get(key) for key in ('index', 'codec_name', 'sample_fmt', 'sample_rate', 'channels', 'channel_layout', 'time_base', 'start_pts', 'duration_ts', 'tags')},
                       'firstPacketPts': first_audio_pts.get(stream['index'])}
                      for stream in data['streams'] if stream['codec_type'] == 'audio'], 'sourceFrameCount': len(frames), 'frames': frames}


def verified_terminal_pts(timeline):
    """Use an observed final-frame or stream duration; never guess a tail."""
    last = timeline['frames'][-1]
    duration = last.get('durationPts')
    if isinstance(duration, int) and not isinstance(duration, bool) and duration > 0:
        return last['pts'] + duration, 'last-frame-duration'
    video = timeline['video']
    if video.get('duration_ts') is not None and video.get('start_pts') is not None:
        end = int(video['start_pts']) + int(video['duration_ts'])
        if end > last['pts']:
            return end, 'stream-integer-duration'
    raise ValueError('Terminal source frame has no proven integer duration; guessed tail timing is refused')


def extract_frames(source, output, ffmpeg, ffprobe, extension='png', fps=0, segments=None):
    """Drop-only sampling keeps real source PTS and never invents duplicate frames."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if extension not in ('png', 'jpg') or float(fps) < 0 or float(fps) > 240:
        raise ValueError('Unsupported extraction format or FPS')
    timeline = probe_timeline(source, ffprobe)
    source_frames = {frame['pts']: frame for frame in timeline['frames']}
    first_time = Fraction(timeline['frames'][0]['pts']) * Fraction(*timeline['frames'][0]['timeBase'])
    selected_frames = []
    for index, segment in enumerate(segments or [None]):
        filters = []
        prefix = f's{index + 1:03d}_' if segment else ''
        if segment:
            start, end = first_time + Fraction(str(segment['start'])), first_time + Fraction(str(segment['end']))
            frame_range = segment.get('sourceFrameRange')
            if frame_range:
                begin, finish = frame_range['startIndex'], frame_range['endIndexExclusive']
                if (frame_range['sourceSha256'] != timeline['source']['sha256']
                    or isinstance(begin, bool) or isinstance(finish, bool) or not isinstance(begin, int) or not isinstance(finish, int)
                    or not 0 <= begin < finish <= len(timeline['frames'])):
                    raise ValueError('Scene segment does not match the original source frame range')
                chosen = timeline['frames'][begin:finish]
                tb = Fraction(*chosen[0]['timeBase'])
                end_pts = timeline['frames'][finish]['pts'] if finish < len(timeline['frames']) else verified_terminal_pts(timeline)[0]
                if (frame_range['timeBase'] != chosen[0]['timeBase'] or frame_range['startPts'] != chosen[0]['pts']
                    or frame_range['endPts'] != end_pts
                    or abs(float((frame_range['startPts'] * tb) - start)) > 1e-9
                    or abs(float((frame_range['endPts'] * tb) - end)) > 1e-9):
                    raise ValueError('Scene segment PTS/seconds disagree; edited boundaries require manual frame selection')
            else:
                chosen = [frame for frame in timeline['frames'] if start <= Fraction(frame['pts']) * Fraction(*frame['timeBase']) < end]
            if not chosen:
                raise ValueError('Selected segment contains no original frames')
            # Select decoded integer frame indices; do not round cut seconds or
            # rely on FFmpeg floating point time comparisons around a boundary.
            filters.append(f"select=gte(n\\,{chosen[0]['sourceFrameIndex']})*lt(n\\,{chosen[-1]['sourceFrameIndex']+1})")
        if float(fps):
            filters.append(f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{1 / float(fps):.12f})")
        filters.extend(('showinfo', 'format=rgb24'))
        with tempfile.TemporaryFile() as log:
            args = [str(ffmpeg), '-hide_banner', '-nostdin', '-copyts', '-i', str(source), '-map', '0:v:0', '-an',
                    '-vf', ','.join(filters), '-fps_mode', 'passthrough', '-start_number', '1',
                    *(['-q:v', '2'] if extension == 'jpg' else []), str(output / (prefix + '%05d.' + extension))]
            result = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=log, timeout=3600)
            log.seek(0)
            pts = []
            errors = []
            for line in log:
                text = line.decode('utf-8', 'replace')
                match = re.search(r'\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:', text)
                if match:
                    pts.append(int(match.group(1)))
                errors.append(text)
                errors = errors[-8:]
            if result.returncode:
                raise RuntimeError('Extraction failed: ' + ''.join(errors))
        images = sorted(output.glob(prefix + '*.' + extension))
        if len(images) != len(pts):
            raise RuntimeError('Extracted images and original PTS observations differ')
        for image, timestamp in zip(images, pts):
            if timestamp not in source_frames:
                raise RuntimeError('Extraction changed source PTS without a declared mapping')
            selected_frames.append({**source_frames[timestamp], 'file': image.name,
                                    'imageSha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'segmentIndex': index})
    if not selected_frames:
        raise ValueError('No frames selected')
    if file_sha256(source) != timeline['source']['sha256']:
        raise RuntimeError('Source video changed during extraction; provenance cannot be published')
    timeline.update({'kind': 'extracted-frames', 'frames': selected_frames,
                     'sampling': {'requestedFps': fps, 'policy': 'original-all-frames' if not fps else 'drop-only-source-pts',
                                  'segments': segments or []}, 'imageCodecLossless': extension == 'png'})
    write_json(output / MANIFEST_NAME, timeline)
    return timeline


def resolve_sequence_timeline(images, reference, ffprobe):
    candidates = [images[0].parent / MANIFEST_NAME, images[0].parent.parent / MANIFEST_NAME]
    for manifest in candidates:
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding='utf-8'))
            if payload.get('schemaVersion') != 1 or payload.get('kind') != 'extracted-frames':
                raise ValueError('Unsupported frame timeline manifest')
            by_stem = {Path(frame['file']).stem: frame for frame in payload['frames']}
            if len(by_stem) != len(payload['frames']):
                raise ValueError('Frame manifest contains duplicate image names')
            if any(image.stem not in by_stem for image in images):
                raise ValueError('Images do not match their source timing manifest; refusing guessed FPS')
            if reference and payload.get('source', {}).get('sha256') != file_sha256(reference):
                raise ValueError('Reference video differs from the frame manifest; refusing mismatched source audio')
            return payload, [by_stem[image.stem] for image in images], 'frame-manifest'
    if reference:
        payload = probe_timeline(reference, ffprobe)
        if len(images) != len(payload['frames']):
            raise ValueError('Reference frame count differs from images; re-extract with a timing manifest')
        if not all(image.stem.isdigit() and int(image.stem) == index for index, image in enumerate(images, 1)):
            raise ValueError('Reference timing requires an uninterrupted original numbered sequence')
        return payload, payload['frames'], 'decoded-reference-pts'
    return None, None, 'explicit-constant-fps'


def rgb_frame_hashes(ffmpeg, source, concat=False):
    args = [ffmpeg, '-v', 'error', '-nostdin']
    if concat:
        args += ['-f', 'concat', '-safe', '0']
    args += ['-i', source, '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
             '-f', 'framehash', '-hash', 'sha256', '-']
    lines = run(args).decode('utf-8').splitlines()
    return [line.split(',')[-1].strip() for line in lines if line and not line.startswith('#')]


def audio_packet_hashes(ffmpeg, source, origin=None, duration=None, codecs=None):
    args = [ffmpeg, '-v', 'error', '-nostdin', '-copyts', '-i', source, '-map', '0:a?', '-vn']
    if codecs:
        args += ['-af', f'atrim=start={float(origin):.12f},asetpts=PTS-{float(origin):.12f}/TB']
        for index, codec in enumerate(codecs):
            args += [f'-c:a:{index}', codec]
    else:
        args += ['-c:a', 'copy']
    if duration is not None:
        args += ['-t', f'{float(duration):.12f}']
    args += ['-f', 'streamhash', '-hash', 'sha256', '-']
    return [line.split(',')[-1].strip() for line in run(args).decode('utf-8').splitlines() if line and not line.startswith('#')]


def audio_decode_timing(ffmpeg, source, origin=None, duration=None):
    """Measure audible decoded-sample timestamps independently of packet hashes."""
    args = [ffmpeg, '-v', 'error', '-nostdin', '-copyts', '-i', source, '-map', '0:a?', '-vn']
    if origin is not None:
        args += ['-af', f'atrim=start={float(origin):.12f},asetpts=PTS-{float(origin):.12f}/TB']
    if duration is not None:
        args += ['-t', f'{float(duration):.12f}']
    args += ['-c:a', 'pcm_s16le', '-f', 'framehash', '-']
    bases, streams = {}, {}
    for line in run(args).decode('utf-8').splitlines():
        match = re.match(r'#tb (\d+): (\d+/\d+)', line)
        if match:
            bases[int(match[1])] = Fraction(match[2])
        elif line and not line.startswith('#'):
            index, _, pts, samples, *_ = [part.strip() for part in line.split(',')]
            index, pts, samples = int(index), int(pts), int(samples)
            stream = streams.setdefault(index, {'start': pts * bases[index], 'end': (pts + samples) * bases[index], 'samples': 0})
            stream['end'] = max(stream['end'], (pts + samples) * bases[index])
            stream['samples'] += samples
    return [{'start': [stream['start'].numerator, stream['start'].denominator],
             'end': [stream['end'].numerator, stream['end'].denominator],
             'sampleCount': stream['samples'], 'sampleTimeBase': [bases[index].numerator, bases[index].denominator]}
            for index, stream in sorted(streams.items())]


def publish_verified_export(output, target, record):
    """Publish the video and its contract together, retaining previous results."""
    import uuid
    output, target = Path(output), Path(target)
    sidecar = Path(str(target) + '.media.json')
    staged_sidecar = output.with_name(output.name + '.media.json')
    write_json(staged_sidecar, record)
    for path in (target, sidecar):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('Export destination must be a regular file, not a link or directory')
    backup = target.parent / ('.video-export-backup-' + uuid.uuid4().hex)
    moved, published = [], []
    try:
        if target.exists() or sidecar.exists():
            backup.mkdir()
        for original in (target, sidecar):
            if original.exists():
                saved = backup / original.name
                original.replace(saved)
                moved.append((original, saved))
        for source, destination in ((output, target), (staged_sidecar, sidecar)):
            source.replace(destination)
            published.append((source, destination))
    except Exception:
        for source, destination in reversed(published):
            destination.replace(source)
        for original, saved in reversed(moved):
            saved.replace(original)
        raise


def encode_sequence(images, target, ffmpeg, ffprobe, reference=None, fps=None, bitrate=16, lossless=False, include_audio=False):
    images = [Path(image) for image in images]
    if not images:
        raise ValueError('No input images')
    if Path(target).resolve() in {image.resolve() for image in images} or (reference and Path(target).resolve() == Path(reference).resolve()):
        raise ValueError('An export cannot replace its source video or input images')
    payload, frames, timing_source = resolve_sequence_timeline(images, reference, ffprobe)
    input_hashes = {image: file_sha256(image) for image in images}
    if frames is None and (fps is None or Fraction(str(fps)) <= 0):
        raise ValueError('An explicit positive constant FPS is required without source timing')
    times = ([Fraction(frame['pts']) * Fraction(*frame['timeBase']) for frame in frames] if frames
             else [Fraction(index, 1) / Fraction(str(fps)) for index in range(len(images))])
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError('Sequence source times overlap or run backwards; select a continuous ordered sequence')
    origin = times[0]
    relative_times = [time - origin for time in times]
    if frames and frames[-1].get('durationPts'):
        last_duration = Fraction(frames[-1]['durationPts']) * Fraction(*frames[-1]['timeBase'])
        final_duration_policy = 'source-decoded-frame-duration'
    elif frames is None:
        last_duration = 1 / Fraction(str(fps))
        final_duration_policy = 'explicit-constant-fps'
    else:
        last_duration = times[-1] - times[-2] if len(times) > 1 else Fraction(1, 25)
        final_duration_policy = 'inferred-last-interval' if len(times) > 1 else 'inferred-single-frame-25fps'
    master = Path(target).suffix.lower() == '.nut' and lossless
    with tempfile.TemporaryDirectory(prefix='.video-encode-', dir=Path(target).parent) as temporary:
        concat = Path(temporary) / 'sequence.ffconcat'
        lines = ['ffconcat version 1.0']
        microseconds = [round(time * 1000000) for time in relative_times]
        for index, image in enumerate(images):
            # ffconcat durations use microseconds; all cumulative PTS are rounded
            # once and their measured error is reported after muxing.
            escaped = image.resolve().as_posix().replace("'", "'\\''")
            lines.extend([f"file '{escaped}'", 'option framerate 1000000'])
            duration_us = microseconds[index + 1] - microseconds[index] if index + 1 < len(times) else round(last_duration * 1000000)
            if duration_us <= 0:
                raise ValueError('Source intervals are below the supported microsecond export precision')
            lines.append(f'duration {duration_us / 1000000:.6f}')
        concat.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        output = Path(temporary) / ('result' + Path(target).suffix)
        args = [ffmpeg, '-v', 'error', '-nostdin', '-copyts', '-f', 'concat', '-safe', '0', '-i', concat]
        has_audio = bool(include_audio and reference and payload and payload.get('audio'))
        complete_source = bool(frames and len(frames) == payload.get('sourceFrameCount') and frames[0]['sourceFrameIndex'] == 0)
        audio_copy = bool(master and complete_source and origin == 0 and payload.get('audio') and
                          all(audio.get('firstPacketPts') is not None and audio['firstPacketPts'] >= 0 for audio in payload['audio']))
        audio_codecs = []
        if has_audio:
            # Audio is trimmed at the original video origin, never normalized by
            # a guessed average FPS. Source PTS are retained in the sidecar.
            args += ['-i', reference]
        args += ['-map', '0:v:0']
        if has_audio:
            args += ['-map', '1:a?']
        args += ['-fps_mode', 'passthrough', '-enc_time_base', '1:1000000']
        if master:
            args += ['-c:v', 'ffv1', '-level', '3', '-slicecrc', '1', '-pix_fmt', 'bgr0', '-color_range', 'pc', '-colorspace', 'rgb']
            output_filter = 'format=bgr0'
            output_parameters = ['range=full', 'colorspace=gbr']
        else:
            source_matrix = payload.get('video', {}).get('color_space') if payload else None
            matrix = source_matrix if source_matrix in ('bt709', 'bt470bg', 'smpte170m', 'smpte240m', 'fcc') else 'bt709'
            scale_matrix = 'bt601' if matrix in ('bt470bg', 'smpte170m') else matrix
            output_filter = f'scale=in_range=full:out_range=limited:out_color_matrix={scale_matrix},format=yuv420p'
            output_parameters = ['range=limited', f'colorspace={matrix}']
            args += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-colorspace', matrix,
                     *(['-crf', '0'] if lossless else ['-b:v', f'{bitrate}M'])]
            if Path(target).suffix.lower() in ('.mp4', '.mov'):
                args += ['-video_track_timescale', '1000000']
        if payload:
            for field, option in (('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')):
                value = payload.get('video', {}).get(field)
                if value and value not in ('unknown', 'unspecified'):
                    args += [option, value]
                    output_parameters.append(f'{"color_trc" if field == "color_transfer" else field}={value}')
        args += ['-vf', output_filter + ',setparams=' + ':'.join(output_parameters)]
        if has_audio:
            if audio_copy:
                args += ['-c:a', 'copy']
            else:
                args += ['-af', f'atrim=start={float(origin):.12f},asetpts=PTS-{float(origin):.12f}/TB']
                if master:
                    # A cut inside an audio packet cannot be copied accurately.
                    # Preserve decoded sample precision in PCM instead of moving
                    # the video timeline to accommodate a negative audio packet.
                    pcm = {'u8': 'pcm_u8', 's16': 'pcm_s16le', 's32': 'pcm_s32le', 's64': 'pcm_s64le', 'flt': 'pcm_f32le', 'dbl': 'pcm_f64le'}
                    for index, audio in enumerate(payload['audio']):
                        sample_format = str(audio.get('sample_fmt', '')).rstrip('p')
                        if sample_format not in pcm:
                            raise ValueError('Unsupported decoded audio precision for a trimmed RGB master')
                        audio_codecs.append(pcm[sample_format])
                        args += [f'-c:a:{index}', pcm[sample_format]]
                else:
                    args += ['-c:a', 'aac', '-b:a', '192k', '-ar', '48000']
        if has_audio and not complete_source:
            args += ['-t', f'{float(relative_times[-1] + last_duration):.12f}']
        args += ['-y', output]
        run(args)
        observed = probe_timeline(output, ffprobe)
        if len(observed['frames']) != len(images):
            raise RuntimeError('Export frame count changed; output was not published')
        errors = [abs(Fraction(frame['pts']) * Fraction(*frame['timeBase']) - expected)
                  for frame, expected in zip(observed['frames'], relative_times)]
        tolerance = max(Fraction(1, 1000000), Fraction(observed['video']['time_base']))
        if max(errors) > tolerance:
            raise RuntimeError('Export PTS differ from recorded timeline beyond container precision')
        rgb_verified = False
        audio_verified = False
        audio_timing_verified = False
        source_audio_timing, output_audio_timing = [], []
        if master:
            rgb_verified = rgb_frame_hashes(ffmpeg, concat, concat=True) == rgb_frame_hashes(ffmpeg, output)
            if not rgb_verified:
                raise RuntimeError('RGB master readback differs from decoded input images; output was not published')
            if has_audio:
                audio_verified = audio_packet_hashes(ffmpeg, reference, origin, None if complete_source else relative_times[-1] + last_duration,
                                                    audio_codecs or None) == audio_packet_hashes(ffmpeg, output)
                if not audio_verified:
                    raise RuntimeError('Master audio packet readback differs from the declared source trim; output was not published')
                source_audio_timing = audio_decode_timing(ffmpeg, reference, origin, None if complete_source else relative_times[-1] + last_duration)
                output_audio_timing = audio_decode_timing(ffmpeg, output)
                audio_timing_verified = len(source_audio_timing) == len(output_audio_timing)
                for expected, actual in zip(source_audio_timing, output_audio_timing):
                    precision = max(Fraction(*expected['sampleTimeBase']), Fraction(*actual['sampleTimeBase']))
                    audio_timing_verified &= expected['sampleCount'] == actual['sampleCount']
                    audio_timing_verified &= all(abs(Fraction(*expected[key]) - Fraction(*actual[key])) <= precision for key in ('start', 'end'))
                if not audio_timing_verified:
                    raise RuntimeError('Master decoded audio timestamps/sample count differ from the declared source trim; output was not published')
        if any(file_sha256(image) != input_hashes[image] for image in images):
            raise RuntimeError('Input images changed during export; output provenance cannot be published')
        if reference and payload and file_sha256(reference) != payload['source']['sha256']:
            raise RuntimeError('Source video changed during export; output provenance cannot be published')
        record = {'schemaVersion': 1, 'product': 'DFL-PT-WEBUI', 'kind': 'video-export', 'mode': 'ffv1-rgb-master' if master else 'h264-codec-lossless-yuv420p' if lossless else 'standard-h264',
                  'timingSource': timing_source, 'sourceOrigin': [origin.numerator, origin.denominator],
                  'finalDurationPolicy': final_duration_policy, 'declaredLastFrameDuration': [last_duration.numerator, last_duration.denominator],
                  'frames': [{**(frame if frames else {}), 'inputFile': image.name, 'inputImageSha256': input_hashes[image], 'outputPts': decoded['pts'], 'outputTimeBase': decoded['timeBase']}
                             for image, frame, decoded in zip(images, frames or [{}] * len(images), observed['frames'])],
                  'verification': {'frameCount': len(images), 'maxPtsErrorSeconds': float(max(errors)), 'ptsToleranceSeconds': float(tolerance),
                                   'outputSha256': file_sha256(output),
                                   'rgbReadbackExact': rgb_verified, 'rgbDomain': 'decoded-input-image-rgb24',
                                   'audioPacketReadbackExact': audio_verified,
                                   'audioDecodedTimingVerified': audio_timing_verified,
                                   'sourceColorAccuracyVerified': False},
                  'sourceColorContract': payload.get('colorContract') if payload else None,
                  'outputColorTags': {field: observed['video'].get(field) for field in COLOR_FIELDS},
                  'audioPolicy': 'original-stream-copy' if audio_copy and has_audio else 'sample-accurate-pcm-at-video-origin' if master and has_audio else 'aac-reencode-trimmed-at-video-origin' if has_audio else 'none',
                  'audioHashDomain': 'source-coded-packets' if audio_copy else 'decoded-samples-in-recorded-pcm-format' if master and has_audio else None,
                  'sourceAudioDecodedTiming': source_audio_timing, 'outputAudioDecodedTiming': output_audio_timing,
                  'sourceAudio': payload.get('audio', []) if payload else [], 'outputAudio': observed['audio']}
        publish_verified_export(output, target, record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--ffprobe', required=True)
    parser.add_argument('--fps', type=float, default=0)
    parser.add_argument('--segments-json')
    arguments = parser.parse_args()
    result = extract_frames(arguments.source, arguments.output, arguments.ffmpeg, arguments.ffprobe,
                            fps=arguments.fps, segments=json.loads(arguments.segments_json) if arguments.segments_json else None)
    print(json.dumps({'schemaVersion': result['schemaVersion'], 'frameCount': len(result['frames']), 'manifest': MANIFEST_NAME}))
