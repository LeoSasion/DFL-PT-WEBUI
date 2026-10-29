"""TAE policy on the existing ME aligned/packed faceset and pair-loader APIs."""
import copy
import hashlib
import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from core.imagelib.warp import gen_warp_params, warp_by_params
from me_backend.data import (AlignedDataset, PairDataLoader as MEPairDataLoader,
                            FACE_TYPES, decode_aligned, read_aligned, _stack_batch)
from me_backend.data_augmentation import transfer_color


def _pretraining_group(reference, metadata):
    if metadata.get('source_id'):
        return 'source-id:' + str(metadata['source_id'])
    person = getattr(reference.packed_sample, 'person_name', None)
    if person:
        return 'person-tag:' + str(person)
    name = str(metadata.get('source_filename') or
               getattr(reference.packed_sample, 'source_filename', None) or reference.path)
    stem = Path(name.replace('\\', '/')).stem
    video = re.fullmatch(r'(.+?)[_-]\d{4,}(?:[_-]\d+)?', stem)
    return ('sequence-name:' + video[1]) if video else 'source-name:' + name


def _pretraining_holdout(group):
    return int(hashlib.sha256(('tae-pretrain-split-v1:' + group).encode()).hexdigest()[:8], 16) % 10000 < 500


@dataclass(frozen=True)
class _TAECheckpoint:
    dataset: object
    epoch: object
    revision: int
    sampler: dict

    def materialize(self):
        data = self.dataset
        if data._checkpoint_epoch is not self.epoch:
            raise RuntimeError('TAE checkpoint snapshot was invalidated by sampler restore')
        if data._exposure_revision == self.revision:
            previous_counts = {}
        elif (data._exposure_revision == self.revision + 1 and
              data._last_exposure_change is not None and
              data._last_exposure_change[0] == self.revision):
            previous_counts = data._last_exposure_change[1]
        else:
            raise RuntimeError('TAE checkpoint snapshot expired; capture after each consumed batch')
        # Only checkpoint boundaries copy the full dataset-sized counter list.
        # One loaded-but-unconsumed batch is undone using its touched indices.
        exposures = list(data.exposures)
        for index, value in previous_counts.items():
            exposures[index] = value
        return {**copy.deepcopy(self.sampler), 'exposures': exposures}


class TAEAlignedDataset(AlignedDataset):
    """Same BGR NCHW 4-tuple as ME, same PAK/ZIP inventory and RNG protocol.

    Random nonlinear warp and optional half-size degradation affect inputs alone;
    RCT/LCT recolors source input and target together against a destination sample. Geometry must
    already match the selected DFL face type and targets are never enlarged.
    """
    def __init__(self, directory, config, source, seed, *, pretraining=False, previous_config=None):
        self.tae_config = config
        native = SimpleNamespace(resolution=config.resolution, face_type=config.face_type,
            batch_size=config.batch_size, ct_mode=config.ct_mode, data_workers=0, uniform_yaw=False,
            allow_large_faceset=pretraining)
        super().__init__(directory, native, source, seed)
        groups, records = {}, []
        self.sample_hashes = []
        def inspect(reference):
            packed = reference.packed_sample
            decoded = decode_aligned(reference, include_metadata=True)
            payload, image, face_type = decoded.payload, decoded.image, decoded.face_type
            metadata = decoded.metadata
            if packed is None:
                source_id = metadata.get("source_id") or metadata.get("source_filename") or reference.path
            else:
                source_id = metadata.get('source_id') or packed.source_filename or reference.path
            if image is None or image.ndim != 3 or image.shape[0] != image.shape[1]:
                raise ValueError(f"{reference.path}: expected a square aligned face")
            if face_type != FACE_TYPES[config.face_type]:
                raise ValueError(f"{reference.path}: face type differs from {config.face_type}; re-extract/resize with DFL first")
            if image.shape[0] < config.resolution:
                raise ValueError(f"{reference.path}: native target {image.shape[0]} is below resolution {config.resolution}; upsampling HQ is disabled")
            # Reuse DFL's landmark/XSeg/priority-mask validation before training.
            _, full, _ = read_aligned(reference, native, decoded=decoded)
            if not np.isfinite(full).all() or np.sum(full) <= 0:
                raise ValueError(f"{reference.path}: empty or invalid face mask")
            sha = hashlib.sha256(payload).hexdigest()
            if pretraining:
                source_id = _pretraining_group(reference, metadata)
            return sha, str(source_id)

        def checked(reference):
            try:
                return inspect(reference), None
            except ValueError as error:
                if not pretraining:
                    raise
                return None, str(error)

        inventory = self._references
        self.pretraining_manifest = []
        self.validation_references = []
        kept, seen, excluded, duplicates = [], set(), 0, 0

        def accept(reference, outcome):
            nonlocal excluded, duplicates
            result, error = outcome
            relative = str(Path(reference.path).relative_to(self.directory))
            if error is not None:
                excluded += 1
                self.pretraining_manifest.append(dict(file=relative, split='excluded', reason=error))
                return
            sha, source_id = result
            if pretraining:
                split = 'duplicate' if sha in seen else 'validation' if _pretraining_holdout(source_id) else 'train'
                self.pretraining_manifest.append(dict(file=relative, sha256=sha, group=source_id, split=split))
                seen.add(sha)
                if split == 'duplicate':
                    duplicates += 1
                    return
                if split == 'validation':
                    self.validation_references.append(reference)
                    return
            index = len(kept)
            kept.append(reference)
            self.sample_hashes.append(sha)
            groups.setdefault(str(source_id), []).append(index)
            records.append((relative, sha, str(source_id)))

        if pretraining:
            # Bound both worker count and pending work; preserve inventory order.
            with ThreadPoolExecutor(max_workers=8) as workers:
                for start in range(0, len(inventory), 256):
                    chunk = inventory[start:start + 256]
                    for reference, outcome in zip(chunk, workers.map(checked, chunk)):
                        accept(reference, outcome)
                    if start % 4096 == 0 or start + 256 >= len(inventory):
                        print(f'[TAE data] checked={min(start+256,len(inventory))}/{len(inventory)} train={len(kept)} holdout={len(self.validation_references)} excluded={excluded} duplicates={duplicates}', flush=True)
        else:
            for reference in inventory:
                accept(reference, checked(reference))
        if not kept:
            raise ValueError('No usable TAE training samples after validation/split')
        if pretraining and not self.validation_references:
            raise ValueError('Pretraining needs a nonempty source-group holdout')
        self._references = kept
        self.paths = [Path(reference.path) for reference in kept]
        self.groups = tuple(tuple(groups[key]) for key in sorted(groups))
        identity = {"records": records, "policy": config.resume_contract(), "version": 1}
        if pretraining:
            identity['pretraining_split'] = 'source-group-sha256-5percent-v1'
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._legacy_fingerprint = None
        self._resume_fingerprints = set()
        if previous_config is not None:
            # Accept exact predecessor policies from before mutable LR and
            # before mutable warp. Records, holdout and immutable policy still
            # participate in every hash; no data-state reset is required.
            eye_policy = {**previous_config.resume_contract(), 'eyes_mouth_weight': previous_config.eyes_mouth_weight}
            previous_policy = {**eye_policy, 'learning_rate': previous_config.learning_rate}
            for policy in (eye_policy, previous_policy, {**previous_policy, 'random_warp': previous_config.random_warp}):
                previous_identity = {**identity, 'policy': policy}
                self._resume_fingerprints.add(hashlib.sha256(json.dumps(previous_identity, sort_keys=True).encode()).hexdigest())
            # v1 fingerprints included batch size and had no ct_mode field.
            old_policy = previous_config.to_dict()
            old_policy.pop('ct_mode', None)
            old_policy.pop('training_mask_mode', None)
            for key in ('decoder_mode', 'decoder_lr_multiplier', 'precision'):
                old_policy.pop(key, None)
            old_identity = {**identity, 'policy': old_policy}
            self._legacy_fingerprint = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
            self._resume_fingerprints.add(self._legacy_fingerprint)
        self.exposures = [0] * len(self._references)
        self._reset_exposure_tracking()
        if pretraining:
            self.pretraining_receipt = dict(pool=str(Path(directory).resolve()), total=len(inventory),
                trainingFaces=len(kept), validationFaces=len(self.validation_references),
                excludedFaces=excluded, duplicateFiles=duplicates, trainingGroups=len(groups),
                fingerprint=self.fingerprint, splitPolicy=identity['pretraining_split'],
                grouping='explicit source_id, packed person tag, sequence filename prefix, otherwise source filename',
                identityIndependentHoldoutVerified=False, duplicates='identical file bytes only')

    def state_dict(self):
        return {**super().state_dict(), "exposures": list(self.exposures)}

    def _reset_exposure_tracking(self):
        self._checkpoint_epoch = object()
        self._exposure_revision = 0
        self._last_exposure_change = None

    def checkpoint_snapshot(self):
        """O(1) bookmark, valid until one further batch (or sampler restore)."""
        return _TAECheckpoint(self, self._checkpoint_epoch, self._exposure_revision,
                              super().state_dict())

    def _exposure_mark(self):
        return self._exposure_revision, self._last_exposure_change

    def _rollback_exposures(self, mark):
        revision, previous_change = mark
        if self._exposure_revision != revision:
            change = self._last_exposure_change
            if self._exposure_revision != revision + 1 or change is None or change[0] != revision:
                raise RuntimeError('TAE exposure rollback is outside the current batch')
            for index, value in change[1].items():
                self.exposures[index] = value
            self._exposure_revision = revision
        self._last_exposure_change = previous_change

    def _commit_batch(self, rng, indexes):
        previous_rng = copy.deepcopy(self.rng.bit_generator.state)
        revision, previous_change = self._exposure_mark()
        previous_counts = {index: self.exposures[index] for index in indexes}
        try:
            self._last_exposure_change = revision, previous_counts
            for index in indexes:
                self.exposures[index] += 1
            self.rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
            self._exposure_revision = revision + 1
        except BaseException:
            # Also handle interruption halfway through counter/RNG commitment.
            for index, value in previous_counts.items():
                self.exposures[index] = value
            self.rng.bit_generator.state = previous_rng
            self._exposure_revision, self._last_exposure_change = revision, previous_change
            raise

    def load_state_dict(self, state):
        exposures = state.get("exposures")
        if not isinstance(exposures, list) or len(exposures) != len(self._references) or any(type(n) is not int or n < 0 for n in exposures):
            raise ValueError("Invalid TAE exposure counters")
        if state.get('fingerprint') in self._resume_fingerprints:
            state = {**state, 'fingerprint': self.fingerprint}
        reference = self._color_reference
        if reference is not None and state.get('reference_fingerprint') in reference._resume_fingerprints:
            state = {**state, 'reference_fingerprint': reference.fingerprint}
        super().load_state_dict(state)
        self.exposures = list(exposures)
        self._reset_exposure_tracking()

    def batch(self):
        if self.closed:
            raise RuntimeError("AlignedDataset is closed")
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        samples, indexes = [], []
        # At most the last aligned sample/reference is retained for this batch.
        # With-replacement sampling can choose it again; never carry decoded
        # pixels across batches or alter the interleaved RNG/read order.
        previous_index = previous_sample = previous_reference = reference_image = None
        for _ in range(self.config.batch_size):
            candidates = self.groups[int(rng.integers(len(self.groups)))] if self.tae_config.source_balanced else range(len(self._references))
            index = int(candidates[int(rng.integers(len(candidates)))])
            indexes.append(index)
            if index != previous_index:
                previous_sample = read_aligned(self._references[index], self.config)
                previous_index = index
            image, full, encoded = previous_sample
            seed, warp_seed = int(rng.integers(2**31)), int(rng.integers(2**31))
            if self.source and self.tae_config.ct_mode in {'rct', 'lct'}:
                if self._color_reference is None:
                    raise ValueError(f'TAE {self.tae_config.ct_mode.upper()} requires a destination color reference')
                references = self._color_reference._references
                reference = references[int(rng.integers(len(references)))]
                if reference is not previous_reference:
                    reference_image = read_aligned(reference, self.config, include_masks=False)[0]
                    previous_reference = reference
                image = transfer_color(image, self.tae_config.ct_mode, reference_image, seed)
            params = gen_warp_params(self.config.resolution, self.tae_config.random_flip,
                rotation_range=[0, 0], scale_range=[0, 0], tx_range=[0, 0], ty_range=[0, 0],
                rnd_state=np.random.RandomState(seed), warp_rnd_state=np.random.RandomState(warp_seed),
                generate_warp=self.tae_config.random_warp)
            def transform(value, warp=False):
                return warp_by_params(params, value, warp, False, True, False, cv2.INTER_LINEAR)
            target = transform(image)
            warped = transform(image, self.tae_config.random_warp)
            if rng.random() < self.tae_config.half_resolution_probability:
                size = self.config.resolution
                warped = cv2.resize(cv2.resize(warped, (size//2, size//2), interpolation=cv2.INTER_AREA),
                                    (size, size), interpolation=cv2.INTER_CUBIC)
            samples.append((np.clip(warped, 0, 1), target, transform(full), transform(encoded)))
        result = _stack_batch(samples)
        if not all(np.isfinite(value).all() for value in result):
            raise ValueError("Nonfinite TAE batch")
        self._commit_batch(rng, indexes)
        return result


class PairDataLoader(MEPairDataLoader):
    def batch(self):
        source_mark = self.source_data._exposure_mark()
        destination_mark = self.destination_data._exposure_mark()
        try:
            return super().batch()
        except BaseException:
            # Native loader rolls back RNG; TAE additionally owns exposure counts.
            # Retain only this pair's touched indices, never copy the whole pool.
            try:
                self.source_data._rollback_exposures(source_mark)
            finally:
                self.destination_data._rollback_exposures(destination_mark)
            raise


def build_pair(src, dst, config, seed, *, pretraining=False, previous_config=None):
    if pretraining and Path(src).resolve() != Path(dst).resolve():
        raise ValueError('Pretraining requires the same SRC and DST pool')
    source = TAEAlignedDataset(src, config, True, seed, pretraining=pretraining, previous_config=previous_config)
    try:
        if pretraining:
            # Reuse immutable validated references, never RNG/exposure state.
            destination = copy.copy(source)
            destination.source = False
            destination._requires_reference = False
            destination.rng = np.random.default_rng(seed + 1)
            destination.exposures = [0] * len(source._references)
            destination._reset_exposure_tracking()
            destination._pending = deque()
        else:
            destination = TAEAlignedDataset(dst, config, False, seed+1, previous_config=previous_config)
        if not pretraining and set(source.sample_hashes) & set(destination.sample_hashes):
            raise ValueError("SRC and DST contain identical files; select distinct identity facesets")
        return PairDataLoader(source, destination)
    except BaseException:
        source.close()
        if "destination" in locals():
            destination.close()
        raise
