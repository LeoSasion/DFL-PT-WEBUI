"""Explicit, checkpoint-bound training configuration."""
from dataclasses import asdict, dataclass, fields
import math

TAE_RESOLUTIONS = (256, 448, 512, 640, 832)
TRAINING_MASK_MODES = ("face", "expanded_feather", "full")


@dataclass
class TrainConfig:
    resolution: int = 256
    face_type: str = "wf"
    inter_dims: int = 256
    hidden_dims: int = 96
    batch_size: int = 1
    precision: str = "bf16"
    decoder_mode: str = "frozen"
    decoder_lr_multiplier: float = 0.1
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    random_warp: bool = True
    random_flip: bool = True
    ct_mode: str = "none"
    half_resolution_probability: float = 0.0
    source_balanced: bool = True
    training_mask_mode: str = "face"
    rgb_weight: float = 1.0
    dssim_weight: float = 1.0
    edge_weight: float = 0.25
    eyes_mouth_weight: float = 1.0
    mask_weight: float = 0.1

    def validate(self):
        for name in ("resolution", "inter_dims", "hidden_dims", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution not in TAE_RESOLUTIONS:
            raise ValueError("TAE resolution must be one of 256, 448, 512, 640, 832")
        if self.inter_dims > 1024 or self.hidden_dims > 1024 or self.batch_size > 64:
            raise ValueError("inter/hidden dims must be <=1024; batch_size must be <=64")
        if self.inter_dims < 4 or self.hidden_dims < 4:
            raise ValueError("inter/hidden dims must be >=4")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.decoder_mode not in {"frozen", "tail", "full"}:
            raise ValueError("decoder_mode must be frozen, tail or full")
        if not math.isfinite(self.decoder_lr_multiplier) or not 0 < self.decoder_lr_multiplier <= 1:
            raise ValueError("decoder_lr_multiplier must be in (0,1]")
        if self.face_type not in {"h", "mf", "f", "wf", "head"}:
            raise ValueError("face_type must be h, mf, f, wf or head")
        for name in ("warmup_steps",):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "rgb_weight", "dssim_weight", "edge_weight", "eyes_mouth_weight", "mask_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rgb_weight + self.dssim_weight + self.edge_weight <= 0 or self.mask_weight <= 0:
            raise ValueError("image and mask supervision must be enabled")
        if not math.isfinite(self.half_resolution_probability) or not 0 <= self.half_resolution_probability <= 1:
            raise ValueError("half_resolution_probability must lie in [0,1]")
        for name in ("random_warp", "random_flip", "source_balanced"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.ct_mode not in {"none", "rct", "lct"}:
            raise ValueError("TAE ct_mode must be none, rct or lct")
        if self.training_mask_mode not in TRAINING_MASK_MODES:
            raise ValueError("TAE training_mask_mode must be face, expanded_feather or full")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("TAE config must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**value).validate()

    def resume_contract(self):
        return {key: value for key, value in self.to_dict().items()
                if key not in {"decoder_mode", "decoder_lr_multiplier", "precision", "batch_size", "ct_mode", "training_mask_mode", "random_warp", "learning_rate", "eyes_mouth_weight"}}

    @property
    def archi(self):
        return "tae-liae"

