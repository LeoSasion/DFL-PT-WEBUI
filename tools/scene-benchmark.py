"""Read-only three-candidate scene comparison and concealed visual review pack.

Supply already acquired, licensed short source clips. The script downloads
nothing, changes no media, and creates no numerical ground truth by inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webui/python'))
from scene_detection import detect, decoded_rgb, file_sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sheet(images, entries, output, title, columns=5):
    from PIL import Image, ImageDraw, ImageFont
    if not entries:
        canvas = Image.new('RGB', (1000, 100), 'white')
        ImageDraw.Draw(canvas).text((15, 20), title + ' | no cuts reported', fill='black')
        canvas.save(output)
        return
    width, height, header = 200, 124, 40
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new('RGB', (width * columns, rows * height + header), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill='black')
    for item, (index, label) in enumerate(entries):
        x, y = (item % columns) * width, (item // columns) * height + header
        canvas.paste(images[index].resize((width, 96)), (x, y))
        draw.text((x + 4, y + 98), label, fill='black')
    canvas.save(output)


def compare(sources, output, ffmpeg, ffprobe):
    if output.exists():
        raise ValueError('Use a fresh output directory; prior review evidence is immutable')
    output.mkdir(parents=True)
    blind = output / 'blind'
    blind.mkdir()
    algorithms = ['ffmpeg', 'adaptive', 'transnetv2']
    random.SystemRandom().shuffle(algorithms)
    mapping = dict(zip('ABC', algorithms))
    key = {'schemaVersion': 1, 'aliases': mapping}
    write(output / 'PRIVATE-REVEAL-KEY.json', key)
    key_sha = file_sha256(output / 'PRIVATE-REVEAL-KEY.json')
    result = {'schemaVersion': 1, 'newTraining': False, 'accuracyGroundTruthAvailable': False,
              'accuracyScoring': 'pending-independent-blinded-review; detector outputs are not ground truth',
              'keySha256': key_sha, 'clips': []}
    for number, source in enumerate(sources, 1):
        before = file_sha256(source)
        clip = {'clip': f'clip-{number}', 'source': {'name': source.name, 'sha256': before, 'bytes': source.stat().st_size},
                'results': {}, 'blindSheets': []}
        for algorithm in algorithms:
            start = time.monotonic()
            value = detect(source, ffmpeg, ffprobe, algorithm)
            if Fraction(value['terminalPts'] - value['firstPts']) * Fraction(*value['timeBase']) > 60:
                raise ValueError('Comparison scope is at most sixty seconds per clip')
            write(output / f'clip-{number}-{algorithm}.json', value)
            clip['results'][algorithm] = {'elapsedSeconds': time.monotonic() - start,
                                         'cutFrameIndices': [cut['sourceFrameIndex'] for cut in value['cuts']],
                                         'sceneCount': value['total'], 'parameters': value['parameters']}
            clip['frameCount'] = value['sourceFrameCount']
            clip['timeBase'], clip['firstPts'], clip['terminalPts'] = value['timeBase'], value['firstPts'], value['terminalPts']
        from PIL import Image
        with decoded_rgb(source, ffmpeg, 200, 96, clip['frameCount']) as frames:
            images = [Image.fromarray(frame.copy()) for _, frame in frames]
        overview = [(index, f'frame {index}') for index in range(0, len(images), 6)]
        overview.append((len(images) - 1, f'frame {len(images) - 1} (end)'))
        path = blind / f'clip-{number}-reference.png'
        sheet(images, overview, path, f'Clip {number}: chronological overview (one per 6 source frames)')
        clip['blindSheets'].append(path.name)
        for alias, algorithm in mapping.items():
            entries = []
            for cut in clip['results'][algorithm]['cutFrameIndices']:
                for index in range(max(0, cut - 2), min(len(images), cut + 3)):
                    entries.append((index, f'{alias} cut {cut} | frame {index}' + (' [CUT]' if index == cut else '')))
            path = blind / f'clip-{number}-candidate-{alias}.png'
            sheet(images, entries, path, f'Clip {number}: concealed candidate {alias}; +/-2 source-frame context')
            clip['blindSheets'].append(path.name)
        if file_sha256(source) != before:
            raise RuntimeError('Original comparison source changed')
        result['clips'].append(clip)
    write(output / 'benchmark.json', result)
    write(blind / 'review-instructions.json', {'schemaVersion': 1, 'keySha256': key_sha,
        'reviewer': 'GPT-6 Astra low (user-selected visual reviewer)',
        'instructions': ['Do not read PRIVATE-REVEAL-KEY.json or algorithm-labelled benchmark results before locking scores.',
            'Inspect the chronological overview first. Record observable actual shot changes, ambiguity, and sample limits.',
            'Inspect each concealed candidate cut with source-frame context. Score false positives, missed cuts and cut-position accuracy.',
            'Candidate no-cut sheets can be correct on static footage; do not reward cut count itself.',
            'No temporal accuracy or general winner claim beyond the visible footage. Animation and static interview coverage are limited.',
            'Save locked scores and ground-truth annotations separately before reveal; exact uncertainty remains explicit.'],
        'clips': [{'clip': clip['clip'], 'source': clip['source'], 'frameCount': clip['frameCount'],
                   'sheets': clip['blindSheets']} for clip in result['clips']]})
    print(json.dumps({'ok': True, 'output': str(output), 'keySha256': key_sha,
                      'accuracyScoring': result['accuracyScoring']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ffmpeg', type=Path, default=ROOT / '_internal/ffmpeg/ffmpeg.exe')
    parser.add_argument('--ffprobe', type=Path, default=ROOT / '_internal/ffmpeg/ffprobe.exe')
    args = parser.parse_args()
    compare([path.resolve(strict=True) for path in args.source], args.output.resolve(), args.ffmpeg, args.ffprobe)
