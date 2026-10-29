"""TAE entry into the existing WebUI controls/heartbeat/preview trainer."""
import math
import json
from pathlib import Path
from datetime import datetime
from me_backend.web_bridge import (TrainingHeartbeat, WebControl, _run_training,
                                   load_config_options, validate_model_name)
from .config import TrainConfig
from .data import build_pair
from .engine import TAEEngine


def train_web(args):
    name = validate_model_name(args.name)
    if args.steps < 0 or args.save_every < 1 or args.preview_every < 1 or args.target_iterations < 0:
        raise ValueError("Invalid step/save/preview/target limits")
    if not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        raise ValueError("TAE max-minutes must be finite and positive")
    stop_at_utc = None
    if getattr(args, 'stop_at_utc', None):
        stop_at_utc = datetime.fromisoformat(args.stop_at_utc.replace('Z', '+00:00'))
        if stop_at_utc.tzinfo is None or stop_at_utc.utcoffset() is None:
            raise ValueError('TAE stop-at-utc requires an explicit timezone')
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be nonnegative")
    directory = Path(args.model).resolve()
    checkpoint = directory / "tae.pt"
    if (directory / "me.pt").exists():
        raise ValueError("Select a separate TAE model directory; this directory contains an ME checkpoint")
    options = load_config_options(args)
    pretraining = bool(getattr(args, "pretraining", False))
    mode = "pretraining" if pretraining else "identity"
    if pretraining and Path(args.src).resolve() != Path(args.dst).resolve():
        raise ValueError("Pretraining requires the same SRC and DST pool")
    if pretraining:
        import cv2
        cv2.setNumThreads(1)
    for key in ("resolution", "batch_size"):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    heartbeat = TrainingHeartbeat(WebControl._path("DFL_WEB_HEARTBEAT_FILE"), 0)
    heartbeat.start()
    pair, completed = None, False
    try:
        with heartbeat.phase("loading-model"):
            if args.resume:
                engine = TAEEngine.load(checkpoint, args.device, weights=args.vae_weights, training=True)
                previous_config = engine.config
                if engine.training_mode != mode:
                    raise ValueError("Resume must preserve pretraining/identity mode")
                requested = TrainConfig.from_dict({**engine.config.to_dict(), **options})
                if engine.reconfigure_decoder(requested):
                    print(f"[TAE new optimization stage] lr={requested.learning_rate} eyes_mouth={requested.eyes_mouth_weight} batch={requested.batch_size} ct={requested.ct_mode} mask={requested.training_mask_mode} warp={requested.random_warp} decoder={requested.decoder_mode}; AdamW and warmup reset; weights, iteration and sampler retained", flush=True)
            else:
                if checkpoint.exists():
                    raise FileExistsError("TAE checkpoint exists; choose Resume")
                engine = TAEEngine(TrainConfig.from_dict(options), args.device, args.seed, args.vae_weights)
                engine.training_mode = mode
        with heartbeat.phase("preparing"):
            engine.pretraining_data = str(Path(args.src).resolve()) if pretraining else None
            pair = build_pair(args.src, args.dst, engine.config, args.seed, pretraining=pretraining,
                              previous_config=previous_config if args.resume else None)
            if pretraining:
                directory.mkdir(parents=True, exist_ok=True)
                from me_backend.web_bridge import atomic_json
                atomic_json(directory / "pretraining-data.json", pair.source_data.pretraining_receipt)
                manifest = directory / "pretraining-manifest.jsonl"
                with manifest.open('w', encoding='utf-8') as stream:
                    for record in pair.source_data.pretraining_manifest:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print('[TAE pretraining data] ' + json.dumps(pair.source_data.pretraining_receipt), flush=True)
            if engine.data_state is not None and not args.reset_data_state:
                pair.load_state_dict(engine.data_state)
        heartbeat.update(iteration=engine.iteration)
        _run_training(args, engine, name, directory, pair, heartbeat, stop_at_utc=stop_at_utc)
        completed = True
    finally:
        try:
            if pair is not None:
                pair.close()
        finally:
            heartbeat.close("finished" if completed else "failed")

