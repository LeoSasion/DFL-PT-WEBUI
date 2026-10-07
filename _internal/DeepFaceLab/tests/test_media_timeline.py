"""Real bounded FFmpeg tests; no neural models or user workspaces."""
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.media_timeline import extract_frames, encode_sequence, probe_timeline, resolve_sequence_timeline
from core.media_timeline import rgb_frame_hashes, publish_verified_export

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = ROOT / 'ffmpeg' / 'ffmpeg.exe'
FFPROBE = ROOT / 'ffmpeg' / 'ffprobe.exe'
pytestmark = pytest.mark.skipif(not FFMPEG.is_file(), reason='project FFmpeg required')


def command(*arguments):
    result = subprocess.run([str(FFMPEG), '-v', 'error', '-nostdin', *map(str, arguments)], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')


@pytest.fixture()
def vfr_source(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    rng = np.random.default_rng(41)
    for index in range(4):
        assert cv2.imwrite(str(inputs / f'{index+1:05d}.png'), rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    concat = tmp_path / 'vfr.ffconcat'
    lines = ['ffconcat version 1.0']
    for index, duration in enumerate((0.04, 0.12, 0.08, 0.04)):
        lines.extend([f"file '{(inputs / f'{index+1:05d}.png').as_posix()}'", 'option framerate 1000', f'duration {duration}'])
    concat.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    source = tmp_path / 'source.nut'
    command('-f', 'concat', '-safe', '0', '-i', concat, '-f', 'lavfi', '-i', 'sine=frequency=500:sample_rate=48000:duration=0.28',
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-c:a', 'pcm_s16le',
            '-enc_time_base', '1:1000', '-fps_mode', 'passthrough', '-t', '0.28', source)
    return source


def test_extraction_preserves_real_vfr_pts_indices_colour_and_audio(vfr_source, tmp_path):
    source = probe_timeline(vfr_source, FFPROBE)
    frames = extract_frames(vfr_source, tmp_path / 'frames', FFMPEG, FFPROBE)
    assert len(frames['frames']) == 4
    assert [row['pts'] for row in frames['frames']] == [row['pts'] for row in source['frames']]
    assert [row['sourceFrameIndex'] for row in frames['frames']] == list(range(4))
    times = [Fraction(row['pts']) * Fraction(*row['timeBase']) for row in frames['frames']]
    assert times == [Fraction(0), Fraction(4, 100), Fraction(16, 100), Fraction(24, 100)]
    assert frames['audio'][0]['codec_name'] == 'pcm_s16le'
    assert frames['colorContract']['sourceColorAccuracyVerified'] is False
    assert all(isinstance(row['pts'], int) for row in frames['frames'])
    assert (tmp_path / 'frames/frames.timeline.json').is_file()


def test_ffv1_rgb_master_roundtrip_keeps_pixels_vfr_and_audio(vfr_source, tmp_path):
    folder = tmp_path / 'frames'
    timeline = extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    images = sorted(folder.glob('*.png'))
    target = tmp_path / 'master.nut'
    report = encode_sequence(images, target, FFMPEG, FFPROBE, reference=vfr_source, lossless=True, include_audio=True)
    assert report['mode'] == 'ffv1-rgb-master'
    assert report['verification']['rgbReadbackExact'] is True
    assert report['verification']['maxPtsErrorSeconds'] <= 0.000001
    assert len(report['frames']) == 4
    assert report['audioPolicy'] == 'original-stream-copy'
    assert report['outputAudio'][0]['codec_name'] == 'pcm_s16le'
    assert report['verification']['audioPacketReadbackExact'] is True
    assert report['verification']['audioDecodedTimingVerified'] is True
    assert report['finalDurationPolicy'] == 'inferred-last-interval'
    original_audio = subprocess.check_output([str(FFMPEG), '-v', 'error', '-i', str(vfr_source), '-map', '0:a:0', '-f', 's16le', '-'])
    output_audio = subprocess.check_output([str(FFMPEG), '-v', 'error', '-i', str(target), '-map', '0:a:0', '-f', 's16le', '-'])
    assert output_audio == original_audio
    assert json.loads(Path(str(target)+'.media.json').read_text())['verification']['rgbReadbackExact']


def test_standard_h264_does_not_claim_rgb_losslessness(vfr_source, tmp_path):
    folder = tmp_path / 'frames'
    extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    report = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'standard.mp4', FFMPEG, FFPROBE,
                             reference=vfr_source, lossless=True, include_audio=True)
    assert report['mode'] == 'h264-codec-lossless-yuv420p'
    assert report['verification']['rgbReadbackExact'] is False
    assert report['outputAudio'][0]['codec_name'] == 'aac'
    assert report['timingSource'] == 'frame-manifest'


def test_drop_only_sampling_and_segment_selection_preserve_source_indices(vfr_source, tmp_path):
    result = extract_frames(vfr_source, tmp_path / 'sampled', FFMPEG, FFPROBE, fps=10,
                            segments=[{'start': 0.03, 'end': 0.25}])
    assert [row['sourceFrameIndex'] for row in result['frames']] == [1, 2]
    assert [row['file'] for row in result['frames']] == ['s001_00001.png', 's001_00002.png']
    assert result['sampling']['policy'] == 'drop-only-source-pts'


def test_missing_frame_mapping_refuses_guessed_reference_fps(vfr_source, tmp_path):
    folder = tmp_path / 'frames'
    extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    image = folder / '00001.png'
    renamed = image.with_name('unknown.png')
    image.rename(renamed)
    with pytest.raises(ValueError, match='manifest'):
        resolve_sequence_timeline([renamed], vfr_source, FFPROBE)


def test_scene_frame_range_keeps_boundary_frame_despite_float_rounding(vfr_source, tmp_path):
    original = probe_timeline(vfr_source, FFPROBE)
    a, b = original['frames'][1:3]
    segment = {'start': 0.04000000000000001, 'end': 0.16,
        'sourceFrameRange': {'sourceSha256': original['source']['sha256'], 'startIndex': 1,
            'endIndexExclusive': 2, 'startPts': a['pts'], 'endPts': b['pts'], 'timeBase': a['timeBase']}}
    result = extract_frames(vfr_source, tmp_path / 'exact-boundary', FFMPEG, FFPROBE, segments=[segment])
    assert [row['sourceFrameIndex'] for row in result['frames']] == [1]
    segment['sourceFrameRange']['sourceSha256'] = '0' * 64
    with pytest.raises(ValueError, match='original source'):
        extract_frames(vfr_source, tmp_path / 'mismatch', FFMPEG, FFPROBE, segments=[segment])


def test_native_cut_preserves_source_and_previous_cut_on_failure(vfr_source, tmp_path):
    from mainscripts.VideoEd import cut_video
    before = vfr_source.read_bytes()
    cut_video(vfr_source, from_time='00:00:00.040', to_time='00:00:00.240', audio_track_id=0, bitrate=2)
    output = vfr_source.with_name('source_cut.nut')
    cut = output.read_bytes()
    observed = probe_timeline(output, FFPROBE)
    assert observed['frames'] and observed['audio'] and observed['video']['codec_name'] == 'h264'
    with pytest.raises(RuntimeError, match='FFmpeg'):
        cut_video(vfr_source, from_time='invalid', to_time='00:00:00.240', audio_track_id=0, bitrate=2)
    assert output.read_bytes() == cut and vfr_source.read_bytes() == before
    assert not list(tmp_path.glob('.cut-*'))


def test_scene_terminal_pts_cannot_be_shortened_to_trim_a_tail_frame(tmp_path):
    from core.media_timeline import verified_terminal_pts
    source = tmp_path / 'tail.mp4'
    command('-f', 'lavfi', '-i', 'color=c=red:s=32x24:r=5:d=0.4', '-c:v', 'libx264', source)
    timeline = probe_timeline(source, FFPROBE)
    begin, terminal = timeline['frames'][1], verified_terminal_pts(timeline)[0]
    tb = Fraction(*begin['timeBase'])
    bad_end = terminal - 1
    segment = {'start': float(begin['pts'] * tb), 'end': float(bad_end * tb), 'sourceFrameRange': {
        'sourceSha256': timeline['source']['sha256'], 'startIndex': 1, 'endIndexExclusive': 2,
        'startPts': begin['pts'], 'endPts': bad_end, 'timeBase': begin['timeBase']}}
    with pytest.raises(ValueError, match='PTS/seconds disagree'):
        extract_frames(source, tmp_path / 'bad-tail', FFMPEG, FFPROBE, segments=[segment])


def test_nonzero_source_origin_remains_in_export_contract(vfr_source, tmp_path):
    shifted = tmp_path / 'shifted.nut'
    command('-i', vfr_source, '-map', '0:v:0', '-map', '0:a:0', '-vf', 'setpts=PTS+0.25/TB',
            '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-c:a', 'copy', '-enc_time_base', '1:1000', '-fps_mode', 'passthrough', shifted)
    folder = tmp_path / 'shifted-frames'
    extracted = extract_frames(shifted, folder, FFMPEG, FFPROBE)
    report = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'shifted-master.nut', FFMPEG, FFPROBE,
                             reference=shifted, lossless=True, include_audio=True)
    assert Fraction(*report['sourceOrigin']) == Fraction(1, 4)
    assert report['verification']['rgbReadbackExact']
    assert report['verification']['audioPacketReadbackExact']
    assert report['frames'][0]['outputPts'] == 0
    assert report['frames'][0]['pts'] == extracted['frames'][0]['pts']


def test_rational_cfr_does_not_accumulate_concat_rounding_error(tmp_path):
    folder = tmp_path / 'cfr'
    folder.mkdir()
    for index in range(100):
        cv2.imwrite(str(folder / f'{index+1:05d}.png'), np.full((16, 16, 3), index, dtype=np.uint8))
    report = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'cfr-master.nut', FFMPEG, FFPROBE,
                             fps='30000/1001', lossless=True)
    assert len(report['frames']) == 100
    assert report['verification']['maxPtsErrorSeconds'] <= 0.000001
    assert report['verification']['rgbReadbackExact']


def test_wrong_reference_source_refuses_audio_mismatch(vfr_source, tmp_path):
    folder = tmp_path / 'frames'
    extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    wrong = tmp_path / 'wrong.nut'
    wrong.write_bytes(vfr_source.read_bytes() + b'different')
    with pytest.raises(ValueError, match='differs'):
        encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'wrong-master.nut', FFMPEG, FFPROBE,
                        reference=wrong, lossless=True, include_audio=True)


def test_aac_priming_does_not_shift_video_master_timestamps(vfr_source, tmp_path):
    compressed = tmp_path / 'source.mp4'
    command('-i', vfr_source, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-fps_mode', 'passthrough', compressed)
    folder = tmp_path / 'aac-frames'
    extract_frames(compressed, folder, FFMPEG, FFPROBE)
    report = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'aac-master.nut', FFMPEG, FFPROBE,
                             reference=compressed, lossless=True, include_audio=True)
    assert report['frames'][0]['outputPts'] == 0
    assert report['verification']['audioPacketReadbackExact']
    assert report['audioPolicy'] == 'sample-accurate-pcm-at-video-origin'
    assert report['outputAudio'][0]['codec_name'] == 'pcm_f32le'


def test_tagged_sdr_matrix_is_recorded_and_hdr_requires_explicit_transform(tmp_path):
    source = tmp_path / 'sdr.mp4'
    command('-f', 'lavfi', '-i', 'testsrc2=size=32x24:rate=5', '-frames:v', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-vf', 'setparams=colorspace=bt709:range=limited:color_primaries=bt709:color_trc=bt709',
            '-colorspace', 'bt709', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', source)
    folder = tmp_path / 'sdr-frames'
    manifest = extract_frames(source, folder, FFMPEG, FFPROBE)
    assert manifest['video']['color_space'] == 'bt709'
    assert manifest['colorContract']['sourceTagsComplete'] is True
    assert manifest['colorContract']['transferConvertedToSrgb'] is False
    master = tmp_path / 'sdr-master.nut'
    report = encode_sequence(sorted(folder.glob('*.png')), master, FFMPEG, FFPROBE, reference=source, lossless=True)
    assert rgb_frame_hashes(FFMPEG, source) == rgb_frame_hashes(FFMPEG, master)
    assert report['verification']['sourceColorAccuracyVerified'] is False
    standard = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'sdr-output.mp4', FFMPEG, FFPROBE, reference=source)
    assert standard['outputColorTags']['color_space'] == 'bt709'
    assert standard['outputColorTags']['color_transfer'] == 'bt709'
    assert standard['outputColorTags']['color_primaries'] == 'bt709'
    hdr = tmp_path / 'hdr.mp4'
    command('-i', source, '-c:v', 'libx264', '-vf', 'setparams=color_trc=smpte2084', '-color_trc', 'smpte2084', hdr)
    with pytest.raises(ValueError, match='HDR'):
        extract_frames(hdr, tmp_path / 'hdr-frames', FFMPEG, FFPROBE)


def test_video_and_contract_publication_roll_back_together(tmp_path, monkeypatch):
    target = tmp_path / 'master.nut'
    sidecar = Path(str(target) + '.media.json')
    target.write_bytes(b'old-video')
    sidecar.write_text('old-contract')
    staged = tmp_path / 'staging'
    staged.mkdir()
    output = staged / 'new.nut'
    output.write_bytes(b'new-video')
    replace = Path.replace
    def fail_contract_publish(self, destination):
        if self.name == 'new.nut.media.json' and Path(destination) == sidecar:
            raise OSError('injected sidecar publication failure')
        return replace(self, destination)
    monkeypatch.setattr(Path, 'replace', fail_contract_publish)
    with pytest.raises(OSError, match='sidecar'):
        publish_verified_export(output, target, {'schemaVersion': 1})
    assert target.read_bytes() == b'old-video'
    assert sidecar.read_text() == 'old-contract'
    assert output.read_bytes() == b'new-video'


def test_export_cannot_replace_source_video(vfr_source, tmp_path):
    folder = tmp_path / 'frames'
    extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    before = vfr_source.read_bytes()
    with pytest.raises(ValueError, match='cannot replace'):
        encode_sequence(sorted(folder.glob('*.png')), vfr_source, FFMPEG, FFPROBE, reference=vfr_source, lossless=True)
    assert vfr_source.read_bytes() == before


def test_delayed_audio_keeps_its_real_offset_from_video(vfr_source, tmp_path):
    source = tmp_path / 'delayed.nut'
    command('-copyts', '-i', vfr_source, '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy',
            '-af', 'asetpts=PTS+0.1/TB', '-c:a', 'pcm_s16le', source)
    folder = tmp_path / 'delayed-frames'
    extract_frames(source, folder, FFMPEG, FFPROBE)
    report = encode_sequence(sorted(folder.glob('*.png')), tmp_path / 'delayed-master.nut', FFMPEG, FFPROBE,
                             reference=source, lossless=True, include_audio=True)
    assert report['verification']['audioDecodedTimingVerified']
    assert Fraction(*report['sourceAudioDecodedTiming'][0]['start']) == Fraction(1, 10)
    assert report['sourceAudioDecodedTiming'] == report['outputAudioDecodedTiming']
    assert report['finalDurationPolicy'] == 'source-decoded-frame-duration'


def test_single_frame_constant_fps_declares_correct_tail(tmp_path):
    folder = tmp_path / 'single'
    folder.mkdir()
    image = folder / '00001.png'
    cv2.imwrite(str(image), np.zeros((16, 16, 3), dtype=np.uint8))
    report = encode_sequence([image], tmp_path / 'single.nut', FFMPEG, FFPROBE, fps=30, lossless=True)
    assert Fraction(*report['declaredLastFrameDuration']) == Fraction(1, 30)
    assert report['finalDurationPolicy'] == 'explicit-constant-fps'


def test_source_changed_during_export_is_not_published(vfr_source, tmp_path, monkeypatch):
    import core.media_timeline as media
    folder = tmp_path / 'changing-frames'
    extract_frames(vfr_source, folder, FFMPEG, FFPROBE)
    target = tmp_path / 'stable.mp4'
    target.write_bytes(b'previous-result')
    probe = media.probe_timeline
    def change_after_output_decode(source, ffprobe):
        result = probe(source, ffprobe)
        if Path(source).name == 'result.mp4':
            vfr_source.write_bytes(vfr_source.read_bytes() + b'changed')
        return result
    monkeypatch.setattr(media, 'probe_timeline', change_after_output_decode)
    with pytest.raises(RuntimeError, match='Source video changed'):
        encode_sequence(sorted(folder.glob('*.png')), target, FFMPEG, FFPROBE, reference=vfr_source)
    assert target.read_bytes() == b'previous-result'


@pytest.fixture()
def denoise_frames(tmp_path):
    inputs = tmp_path / 'noisy-inputs'
    inputs.mkdir()
    rng = np.random.default_rng(19)
    for index in range(3):
        assert cv2.imwrite(str(inputs / f'{index+1:05d}.png'),
                           rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    source = tmp_path / 'noisy.nut'
    command('-framerate', '25', '-i', inputs / '%05d.png', '-c:v', 'ffv1',
            '-pix_fmt', 'bgr0', source)
    folder = tmp_path / 'denoised-frames'
    extract_frames(source, folder, FFMPEG, FFPROBE)
    return source, folder


def test_denoise_updates_hashes_keeps_original_provenance_and_merge_timing(denoise_frames, tmp_path):
    from mainscripts.VideoEd import denoise_image_sequence
    from core.media_timeline import file_sha256
    from merger.temporal_geometry import bind_timeline

    source, folder = denoise_frames
    manifest_path = folder / 'frames.timeline.json'
    original = json.loads(manifest_path.read_text())
    images = sorted(folder.glob('*.png'))
    before_hashes = {image.name: file_sha256(image) for image in images}
    receipt = denoise_image_sequence(folder, factor=7)
    updated = json.loads(manifest_path.read_text())
    timings, audit = bind_timeline(images)
    assert audit['status'] == 'bound'
    assert updated['source'] == original['source']
    assert file_sha256(source) == original['source']['sha256']
    assert [row['pts'] for row in timings] == [row['pts'] for row in original['frames']]
    assert [row['sourceFrameIndex'] for row in timings] == [0, 1, 2]
    assert all(row['originalImageSha256'] == before_hashes[row['file']]
               and row['imageSha256'] == file_sha256(folder / row['file']) for row in updated['frames'])
    history = updated['processingHistory'][-1]
    assert history['sourcePtsPreserved'] and history['parameters']['lumaSpatial'] == 7
    assert all(row['inputImageSha256'] == before_hashes[row['file']]
               and row['outputImageSha256'] != row['inputImageSha256'] for row in history['frames'])
    assert len(receipt['entries']) == 4 and receipt['state'] == 'committed'
    for entry in receipt['entries']:
        assert file_sha256(Path(receipt['archive_path']) / entry['backup']) == entry['source_sha256']
    report = encode_sequence(images, tmp_path / 'denoised-master.nut', FFMPEG, FFPROBE,
                             reference=source, lossless=True)
    assert report['verification']['rgbReadbackExact'] and len(report['frames']) == 3
    denoise_image_sequence(folder, factor=3)
    repeated = json.loads(manifest_path.read_text())
    assert len(repeated['processingHistory']) == 2
    assert all(row['originalImageSha256'] == before_hashes[row['file']] for row in repeated['frames'])
    assert bind_timeline(images)[1]['status'] == 'bound'


def test_denoise_rolls_back_images_and_manifest_on_publication_failure(denoise_frames, monkeypatch):
    from mainscripts.VideoEd import denoise_image_sequence
    from core import faceset_transaction as transaction

    _, folder = denoise_frames
    originals = {path.name: path.read_bytes() for path in folder.iterdir()}
    publish = transaction._exclusive_publish
    failed = False
    def fail_manifest_once(source, target):
        nonlocal failed
        if Path(target).name == 'frames.timeline.json' and not failed:
            failed = True
            raise OSError('injected denoise manifest publication failure')
        return publish(source, target)
    monkeypatch.setattr(transaction, '_exclusive_publish', fail_manifest_once)
    with pytest.raises(OSError, match='manifest publication failure'):
        denoise_image_sequence(folder, factor=7)
    assert failed
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == originals
    receipts = list(folder.parent.glob(folder.name + '_history/faceset-*/receipt.json'))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())['state'] == 'rolled_back'


def test_denoise_refuses_already_stale_frame_manifest(denoise_frames):
    from mainscripts.VideoEd import denoise_image_sequence

    _, folder = denoise_frames
    first = sorted(folder.glob('*.png'))[0]
    assert cv2.imwrite(str(first), np.zeros((64, 64, 3), np.uint8))
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(ValueError, match='differs from its timing manifest'):
        denoise_image_sequence(folder, factor=7)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert not (folder.parent / (folder.name + '_history')).exists()
