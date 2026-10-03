"""Local ME training bridge shared by the CLI and fixed WebUI command registry."""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import cv2
import numpy as np
import torch

from .config import MEConfig
from .checkpoint_backup import create_backup
from .data import AlignedDataset, read_aligned
from .engine import MEEngine
from .loss_console import LossConsole


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            # Windows readers can briefly deny replacement of the destination.
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32) or attempt == 9:
                raise
            time.sleep(0.02 * (attempt + 1))


def atomic_image(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, np.round(np.clip(value, 0, 1) * 255).astype(np.uint8))
    if not ok:
        raise IOError(f'Cannot encode {path}')
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    encoded.tofile(temporary)
    os.replace(temporary, path)


def validate_model_name(name):
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError('ME model name must contain 1..64 characters')
    if name != name.strip() or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith(('.', ' ')):
        raise ValueError('Invalid ME model name')
    if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', name, re.I):
        raise ValueError('Reserved Windows model name')
    return name


class WebControl:
    OPERATIONS = {'save', 'backup', 'preview', 'evaluate', 'close'}

    def __init__(self):
        self.control = self._path('DFL_WEB_CONTROL_FILE')
        self.preview = self._path('DFL_WEB_PREVIEW_FILE')
        self.ack = self._path('DFL_WEB_CONTROL_ACK_FILE')
        self.offset = 0
        self.pending = b''

    @staticmethod
    def _path(name):
        value = os.environ.get(name, '').strip()
        return Path(value) if value else None

    def poll(self):
        if self.control is None or not self.control.exists():
            return []
        with self.control.open('rb') as stream:
            stream.seek(self.offset)
            new = stream.read(1024 * 1024)
            self.offset = stream.tell()
        lines = (self.pending + new).split(b'\n')
        self.pending = lines.pop()
        if len(self.pending) > 8192:
            raise ValueError('Oversized control request')
        requests = []
        for line in lines:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or request.get('operation') not in self.OPERATIONS:
                    raise ValueError('Unknown control operation')
                requests.append(request)
            except (ValueError, UnicodeError) as error:
                print(f'[ME control] rejected request: {error}', flush=True)
        return requests

    def complete(self, request, engine, checkpoint, error=None):
        if self.ack is not None:
            atomic_json(self.ack, dict(operation=request['operation'],
                requestedAt=request.get('requestedAt'), status='failed' if error else 'completed',
                iteration=engine.iteration, checkpoint=str(checkpoint.resolve()),
                completedAt=utc_now(), error=str(error) if error else None))


class TrainingHeartbeat:
    """Process identity, liveness, and progress survive the Web server process."""

    def __init__(self, target, iteration):
        self.target = target
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error_log_at = 0.0
        now = utc_now()
        self.state = dict(pid=os.getpid(), startedAt=now, at=now,
                          progressAt=now, phaseAt=now, iteration=iteration, phase='starting')

    def _write_locked(self):
        if self.target is not None:
            self.state['at'] = utc_now()
            try:
                atomic_json(self.target, self.state)
            except OSError as error:
                # Heartbeat is an observer. A transient file-sharing error must
                # make the UI show an unknown/stale state, never kill training.
                now = time.monotonic()
                if now - self.last_error_log_at >= 30:
                    print(f'[ME heartbeat] failed: {error}', flush=True)
                    self.last_error_log_at = now

    def start(self):
        if self.target is None:
            return
        with self.lock:
            self._write_locked()

        def pulse():
            while not self.stop_event.wait(5):
                try:
                    with self.lock:
                        self._write_locked()
                except OSError as error:
                    print(f'[ME heartbeat] failed: {error}', flush=True)
        self.thread = threading.Thread(target=pulse, name='me-heartbeat', daemon=True)
        self.thread.start()

    def update(self, phase=None, iteration=None, progressed=False):
        if self.target is None:
            return
        with self.lock:
            if phase is not None:
                if phase != self.state['phase']:
                    self.state['phaseAt'] = utc_now()
                self.state['phase'] = phase
            if iteration is not None:
                self.state['iteration'] = iteration
            if progressed:
                self.state['progressAt'] = utc_now()
            self._write_locked()

    @contextmanager
    def phase(self, name):
        if self.target is None:
            yield
            return
        with self.lock:
            previous = self.state['phase']
        self.update(phase=name)
        try:
            yield
        finally:
            self.update(phase=previous)

    def close(self, phase):
        if self.target is None:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
        self.update(phase=phase)


@torch.inference_mode()
def make_preview(engine, source, destination):
    engine.network.eval()
    outputs = engine.network(engine.tensors(source)[1], engine.tensors(destination)[1])
    def nhwc(tensor):
        return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    src = source[1].transpose(0, 2, 3, 1)
    dst = destination[1].transpose(0, 2, 3, 1)
    rows = []
    mask = nhwc(outputs['swap_mask'] * outputs['dst_mask'])
    for i in range(min(len(src), 4)):
        rows.append(np.concatenate((src[i], nhwc(outputs['src'])[i], dst[i],
            nhwc(outputs['dst'])[i], nhwc(outputs['swap'])[i], np.repeat(mask[i], 3, axis=-1)), axis=1))
    return np.concatenate(rows, axis=0)


def network_weights_sha256(network):
    """Hash generator weights independent of state-dict order and device.

    Copy at most one tensor to CPU at a time, so evaluation does not need a
    second full model-sized buffer when the network lives on a GPU.
    """
    digest = hashlib.sha256(b'ME-network-weights-v1\0')
    for name, tensor in sorted(network.state_dict().items()):
        metadata = json.dumps({
            'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
        }, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        raw = tensor.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(len(metadata).to_bytes(8, 'big'))
        digest.update(metadata)
        digest.update(raw.nbytes.to_bytes(8, 'big'))
        digest.update(memoryview(raw))
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(engine, name, src_directory, dst_directory):
    helper_directory = os.environ.get('DFL_WEBUI_PYTHON')
    if not helper_directory:
        helper_directory = str(Path(__file__).resolve().parents[3] / 'webui' / 'python')
    if helper_directory not in sys.path:
        sys.path.insert(0, helper_directory)
    from training_evaluation import (AtomicEvaluationSnapshot, load_evaluation_manifest,
        reconstruction_metrics, swap_metrics)
    manifest_path = os.environ.get('DFL_WEB_EVAL_MANIFEST')
    root = os.environ.get('DFL_WEB_EVAL_ROOT')
    model_key = os.environ.get('DFL_WEB_EVAL_MODEL_KEY')
    if not all((manifest_path, root, model_key)):
        raise ValueError('Evaluation manifest is not configured for this training job')
    evaluation_src = os.environ.get('DFL_WEB_EVAL_SRC')
    evaluation_dst = os.environ.get('DFL_WEB_EVAL_DST')
    if bool(evaluation_src) != bool(evaluation_dst):
        raise ValueError('Evaluation SRC and DST facesets must be configured together')
    try:
        timeout_seconds = int(os.environ.get('DFL_WEB_EVAL_TIMEOUT_SECONDS', '120'))
    except ValueError as error:
        raise ValueError('Evaluation timeout must be 1..3600 seconds') from error
    if not 1 <= timeout_seconds <= 3600:
        raise ValueError('Evaluation timeout must be 1..3600 seconds')
    manifest, samples = load_evaluation_manifest(manifest_path, root, model_key,
        {'src': evaluation_src or src_directory, 'dst': evaluation_dst or dst_directory})
    if manifest['modelName'] != name or manifest['modelClass'] != 'ME':
        raise ValueError('Evaluation manifest does not select this ME model')
    signature = dict(modelClass='ME', archi=engine.config.archi,
        resolution=engine.config.resolution, faceType=engine.config.face_type, dataFormat='NCHW')
    weights_sha256 = network_weights_sha256(engine.network)
    snapshot = AtomicEvaluationSnapshot(root, model_key, manifest['manifestId'], engine.iteration,
        signature, model_weights_sha256=weights_sha256)
    deadline = time.monotonic() + timeout_seconds
    try:
        engine.network.eval()
        for side in ('src', 'dst'):
            for sample in samples[side]:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'ME evaluation exceeded {timeout_seconds} seconds')
                image, full, encoded = read_aligned(sample['path'], engine.config)
                x = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)[None]), device=engine.device)
                outputs = engine.network(x, x)
                def first(key):
                    return outputs[key][0].cpu().numpy().transpose(1, 2, 0)
                reconstruction = first(side)
                prediction_mask = first(side + '_mask')
                metrics = reconstruction_metrics(image, reconstruction, full,
                    np.clip(encoded - 1, 0, 1), prediction_mask)
                variants = dict(input=image, reconstruction=reconstruction,
                    **{'target-mask': full, 'predicted-mask': prediction_mask})
                if side == 'dst':
                    swap = first('swap')
                    variants['swap'] = swap
                    metrics['swap'] = swap_metrics(image, swap, full, first('swap_mask'))
                snapshot.add_sample(sample, variants, metrics)
        summary = snapshot.publish()
    except BaseException:
        snapshot.abort()
        raise
    print(f"[ME evaluation] snapshot {summary['snapshotId']} completed", flush=True)
    return summary


def load_config_options(args):
    """Read either CLI configuration source, including older bridge Namespaces."""
    config_file = getattr(args, 'config', None)
    config_json = getattr(args, 'config_json', None)
    if config_file is not None and config_json is not None:
        raise ValueError('Use only one of --config and --config-json')
    if config_file is not None:
        options = json.loads(Path(config_file).read_text(encoding='utf-8-sig'))
    elif config_json is not None:
        options = json.loads(config_json)
    else:
        return {}
    if not isinstance(options, dict):
        raise ValueError('ME config must be a JSON object')
    return options


def load_training_engine(args):
    """Strict resume by default; training changes require an explicit flag."""
    checkpoint = Path(args.model).resolve() / 'me.pt'
    options = load_config_options(args)
    options.update({field: getattr(args, field) for field in ('archi', 'use_rg', 'use_fp16', 'optimizer_on_cpu', 'pretrain', 'resolution', 'batch_size')
                    if getattr(args, field, None) is not None})
    if args.resume:
        if getattr(args, 'initialize_from', None):
            raise ValueError('Cannot initialize pretrained weights while resuming')
        engine = MEEngine.load(checkpoint, args.device)
        requested = MEConfig.from_dict({**engine.config.to_dict(), **options})
        changed = engine.config.changed_fields(requested)
        if changed and not getattr(args, 'allow_config_change', False):
            raise ValueError('Resume config differs from the checkpoint; use --allow-config-change explicitly: '+', '.join(sorted(changed)))
        if changed or getattr(args, 'reset_optimizer', False):
            engine.reconfigure(requested, getattr(args, 'reset_optimizer', False))
            print('[ME config changed] '+', '.join(sorted(changed)), flush=True)
        return engine
    if checkpoint.exists():
        raise FileExistsError('ME checkpoint already exists; select Resume')
    initialize = getattr(args, 'initialize_from', None)
    pretrained = None
    if initialize:
        pretrained_path = Path(initialize)
        if pretrained_path.is_dir():
            pretrained_path = pretrained_path / 'me.pt'
        pretrained = MEEngine.load(pretrained_path, 'cpu')
        options = {**pretrained.config.to_dict(), 'pretrain': False, **options}
    config = MEConfig.from_dict(options)
    engine = MEEngine(config, args.device, args.seed)
    if pretrained is not None:
        config.assert_compatible_network(pretrained.config)
        engine.network.load_state_dict(pretrained.network.state_dict(), strict=True)
        print('[ME initialized] network weights imported; optimizer and iteration start at zero', flush=True)
    return engine


def train_web(args):
    name = validate_model_name(args.name)
    if args.steps < 0 or args.save_every < 1 or args.preview_every < 1:
        raise ValueError('steps must be nonnegative; save/preview intervals must be positive')
    backup_every = getattr(args, 'backup_every', 1000)
    backup_keep = getattr(args, 'backup_keep', 3)
    if backup_every < 1 or not 2 <= backup_keep <= 32:
        raise ValueError('backup interval must be positive and retention must keep 2..32 generations')
    if getattr(args, 'target_iterations', 0) < 0:
        raise ValueError('target-iterations must be nonnegative')
    directory = Path(args.model).resolve()
    engine = load_training_engine(args)
    heartbeat = TrainingHeartbeat(WebControl._path('DFL_WEB_HEARTBEAT_FILE'), engine.iteration)
    heartbeat.start()
    source_path, destination_path = args.src, args.dst
    if engine.config.pretrain:
        pretraining = getattr(args, 'pretraining_data_dir', None)
        if not pretraining:
            raise ValueError('Pretraining requires --pretraining-data-dir')
        source_path = destination_path = pretraining
    source_data = destination_data = pair = None
    completed = False
    try:
        from .data import PairDataLoader
        source_data = AlignedDataset(source_path, engine.training_config, True, args.seed)
        destination_data = AlignedDataset(destination_path, engine.training_config, False, args.seed + 1)
        if engine.data_state and not args.reset_data_state:
            source_data.load_state_dict(engine.data_state['src'])
            destination_data.load_state_dict(engine.data_state['dst'])
        if args.reset_data_state:
            from .replay import HardSampleReplay
            engine.replay = HardSampleReplay(engine.config.retraining_capacity)
        pair = PairDataLoader(source_data, destination_data)
        _run_training(args, engine, name, directory, pair, heartbeat)
        completed = True
    finally:
        try:
            if pair is not None:
                pair.close()
            else:
                for dataset in (source_data, destination_data):
                    if dataset is not None:
                        dataset.close()
        finally:
            heartbeat.close('finished' if completed else 'failed')


def _run_training(args, engine, name, directory, pair, heartbeat):
    checkpoint = directory / 'me.pt'
    # Validate actual image/mask decoding before publishing a new model.
    with heartbeat.phase('preparing'):
        next_data_state = pair.state_dict()
        source, destination = pair.batch()
    if getattr(args, 'debug_samples', False):
        def sample_grid(batch):
            rows = []
            for index in range(min(len(batch[0]), 4)):
                panels = [batch[0][index], batch[1][index], np.repeat(batch[2][index], 3, axis=0),
                          np.repeat(batch[3][index] / 3., 3, axis=0)]
                rows.append(np.concatenate(panels, axis=2).transpose(1, 2, 0))
            return np.concatenate(rows, axis=0)
        source_grid, destination_grid = sample_grid(source), sample_grid(destination)
        atomic_image(directory / 'debug-src.png', source_grid)
        atomic_image(directory / 'debug-dst.png', destination_grid)
        preview_path = WebControl._path('DFL_WEB_PREVIEW_FILE')
        if preview_path is not None:
            atomic_image(preview_path, np.concatenate([source_grid, destination_grid], axis=0))
        print('[ME sample debug] wrote warped/target/full/priority-mask grids; no optimizer update', flush=True)
        return
    controls = WebControl()
    directory.mkdir(parents=True, exist_ok=True)
    history_path = directory / 'loss-history.jsonl'
    loss_console = LossConsole()

    def save():
        loss_console.flush_latest()
        with heartbeat.phase('saving'):
            engine.save(checkpoint, next_data_state)
            atomic_json(directory / 'metadata.json', dict(format='me-pytorch', version=1,
                schemaVersion=1, name=name, model_class='ME', modelClass='ME', iteration=engine.iteration,
                config=engine.config.to_dict(), checkpoint='me.pt', savedAt=utc_now()))
            print(f'[ME saved] iteration={engine.iteration} checkpoint={checkpoint}', flush=True)

    def backup(kind):
        loss_console.finish_line()
        with heartbeat.phase('backup'):
            result = create_backup(directory, engine.iteration, kind=kind,
                                   keep=getattr(args, 'backup_keep', 3))
            print(f"[ME backup] {result['id']} iteration={engine.iteration}", flush=True)

    def preview():
        with heartbeat.phase('preview'):
            value = make_preview(engine, source, destination)
            atomic_image(directory / 'preview.png', value)
            if controls.preview is not None:
                atomic_image(controls.preview, value)

    print(f"[ME PyTorch] device={engine.device} model={name} resolution={engine.config.resolution} batch={engine.config.batch_size}", flush=True)
    print(f'[ME resume] iteration={engine.iteration}', flush=True)
    save()
    preview()
    heartbeat.update(phase='training')
    completed = 0
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous_interrupt = signal.signal(signal.SIGINT, request_stop)
    try:
        while (args.steps == 0 or completed < args.steps) and (not getattr(args, 'target_iterations', 0) or engine.iteration < args.target_iterations):
            if stopping:
                save()
                preview()
                print('[ME stopped] checkpoint saved successfully', flush=True)
                return
            for request in controls.poll():
                operation = request['operation']
                try:
                    if operation in ('save', 'close', 'backup'):
                        save()
                    if operation == 'backup':
                        backup('manual')
                    if operation in ('preview', 'close'):
                        preview()
                    if operation == 'evaluate':
                        loss_console.finish_line()
                        with heartbeat.phase('evaluate'):
                            evaluate(engine, name, args.src, args.dst)
                    controls.complete(request, engine, checkpoint)
                except Exception as error:
                    controls.complete(request, engine, checkpoint, error)
                    if operation in ('save', 'close'):
                        raise
                    print(f'[ME {operation}] failed: {error}', flush=True)
                if operation == 'close':
                    print('[ME stopped] checkpoint saved successfully', flush=True)
                    return
            start = time.perf_counter()
            result = engine.train_step(source, destination)
            elapsed = time.perf_counter() - start
            completed += 1
            heartbeat.update(iteration=engine.iteration, progressed=True)
            next_data_state = pair.state_dict()
            with history_path.open('a', encoding='utf-8') as history:
                history.write(json.dumps({**result, 'seconds': elapsed, 'at': utc_now()}) + '\n')
            loss_console.update(engine.iteration, elapsed, result['src_loss'], result['dst_loss'])
            automatic_backup = engine.iteration % getattr(args, 'backup_every', 1000) == 0
            if engine.iteration % args.save_every == 0 or automatic_backup:
                save()
            if automatic_backup:
                try:
                    backup('automatic')
                except Exception as error:
                    # The primary checkpoint is already durable. Do not discard
                    # an otherwise healthy long run because rotation failed.
                    print(f'[ME automatic backup] failed: {error}', flush=True)
            if engine.iteration % args.preview_every == 0:
                preview()
            with heartbeat.phase('loading'):
                source, destination = pair.batch()
    except KeyboardInterrupt:
        loss_console.flush_latest()
        print('[ME stopping] saving current state after interrupt', flush=True)
        save()
        preview()
        print('[ME stopped] checkpoint saved successfully', flush=True)
        return
    finally:
        loss_console.finish_line()
        signal.signal(signal.SIGINT, previous_interrupt)
    save()
    preview()
