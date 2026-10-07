"""PyTorch-only face descriptors and opt-in evaluation adapters.

SFace's pinned ONNX file is a weight container, never an execution runtime.
All adapters consume the same already aligned 112x112 RGB crop. Descriptors
are anonymous similarity features; a score alone does not establish identity.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vision_assets import canonical_digest, resolve_asset_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / "workspace/.vision-models/identity"
SFACE_PATH = ROOT / "_internal/vision_models/face_recognition_sface_2021dec.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
IDENTITY_CONTRACTS = {
    "AdaFace": {
        "official_source": "https://github.com/mk-minchul/AdaFace",
        "revision": "c60eaa786a42c03444f3df7096dbaf9d57ae010d",
        "execution_source": "source/net.py",
        "weight_path": "weights/adaface_ir101_webface12m.ckpt",
        "weight_sha256": "0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0",
        "source_digest": "1aa953d7b4bfcf3dba28c37bbe664b6b2ab19856fb110fe0747527057975802f",
    },
    "MagFace": {
        "official_source": "https://github.com/IrvingMeng/MagFace",
        "revision": "99bae614ac2643b9694bf18e0c5645272ff6acfa",
        "execution_source": "source/models/iresnet.py",
        "weight_path": "weights/magface_iresnet100.pth",
        "weight_sha256": "cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58",
        "source_digest": "a08aa2f73d301bf03e334dab453f2c817d3a66a8a6709888433edd60b38d23f6",
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_crop(rgb):
    crop = np.asarray(rgb)
    if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
        raise ValueError("Expected a shared aligned 112x112 RGB uint8 crop")
    return crop


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid identity descriptor")
    return vector / norm


class SFaceTorch(nn.Module):
    """Strict executor for the 88-node published SFace inference graph.

    The digest limits this interpreter to the audited graph. Unsupported
    operators, attributes, graph branches or graph inputs fail closed.
    """
    def __init__(self, path=SFACE_PATH, device="cpu"):
        super().__init__()
        if file_sha256(path) != SFACE_SHA256:
            raise ValueError("SFace model SHA256 mismatch")
        import onnx
        from onnx import numpy_helper, helper
        graph = onnx.load(str(path)).graph
        supported = {
            "Sub": set(), "Mul": set(), "PRelu": set(),
            "Conv": {"dilations", "group", "kernel_shape", "pads", "strides"},
            "BatchNormalization": {"epsilon", "momentum"},
            "Dropout": {"ratio"}, "Flatten": {"axis"},
            "Gemm": {"alpha", "beta", "transA", "transB"},
        }
        self.weight_names = {}
        for index, initializer in enumerate(graph.initializer):
            name = f"weight_{index}"
            self.register_buffer(name, torch.from_numpy(numpy_helper.to_array(initializer).copy()))
            self.weight_names[initializer.name] = name
        inputs = [item.name for item in graph.input if item.name not in self.weight_names]
        if inputs != ["data"] or len(graph.output) != 1 or graph.output[0].name != "fc1":
            raise ValueError("Unsupported SFace graph interface")
        self.nodes = []
        known = set(self.weight_names) | {"data"}
        for node in graph.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.domain or node.op_type not in supported or not set(attrs) <= supported[node.op_type]:
                raise ValueError(f"Unsupported SFace operator: {node.op_type}")
            if len(node.output) != 1 or any(name not in known for name in node.input):
                raise ValueError("Unsupported SFace graph topology")
            self.nodes.append((node.op_type, tuple(node.input), node.output[0], attrs))
            known.add(node.output[0])
        self.to(device).eval()

    def forward(self, blob):
        if blob.ndim != 4 or tuple(blob.shape[1:]) != (3, 112, 112):
            raise ValueError("Expected SFace NCHW RGB 112x112 input")
        values = {name: getattr(self, buffer) for name, buffer in self.weight_names.items()}
        values["data"] = blob
        for op, names, output, attrs in self.nodes:
            args = [values[name] for name in names]
            if op == "Sub":
                value = args[0] - args[1]
            elif op == "Mul":
                value = args[0] * args[1]
            elif op == "Conv":
                pads = attrs.get("pads", [0, 0, 0, 0])
                if pads[:2] != pads[2:]:
                    raise ValueError("Unsupported asymmetric SFace padding")
                value = F.conv2d(args[0], args[1], args[2] if len(args) == 3 else None,
                                 stride=attrs.get("strides", [1, 1]), padding=pads[:2],
                                 dilation=attrs.get("dilations", [1, 1]), groups=attrs.get("group", 1))
            elif op == "BatchNormalization":
                value = F.batch_norm(args[0], args[3], args[4], args[1], args[2],
                                     training=False, eps=attrs.get("epsilon", 1e-5))
            elif op == "PRelu":
                value = torch.where(args[0] >= 0, args[0], args[0] * args[1])
            elif op == "Dropout":
                value = args[0]  # The published inference graph has no training input.
            elif op == "Flatten":
                axis = attrs.get("axis", 1)
                value = args[0].reshape(int(np.prod(args[0].shape[:axis])), -1)
            elif op == "Gemm":
                left = args[0].T if attrs.get("transA", 0) else args[0]
                right = args[1].T if attrs.get("transB", 0) else args[1]
                value = attrs.get("alpha", 1) * (left @ right)
                if len(args) == 3:
                    value = value + attrs.get("beta", 1) * args[2]
            values[output] = value
        return values["fc1"]

    @torch.inference_mode()
    def embedding_bgr(self, face):
        face = validate_crop(face)
        blob = torch.from_numpy(face[:, :, ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0)
        return self(blob.to(next(self.buffers()).device)).cpu().numpy().reshape(-1)


def verified_identity(directory, expected_model=None):
    directory = Path(directory)
    identity = json.loads(resolve_asset_path(directory, "identity.json").read_text(encoding="utf-8"))
    family = expected_model or identity.get("model")
    if family not in IDENTITY_CONTRACTS or identity.get("model") != family:
        raise ValueError("Identity manifest model does not match the fixed adapter")
    contract = IDENTITY_CONTRACTS[family]
    for field in ("official_source", "revision"):
        if identity.get(field) != contract[field]:
            raise ValueError(f"Identity manifest differs from fixed source: {field}")
    weights = identity.get("weights", [])
    if len(weights) != 1 or weights[0].get("path") != contract["weight_path"] or weights[0].get("sha256") != contract["weight_sha256"]:
        raise ValueError("Identity checkpoint differs from fixed model contract")
    if contract["execution_source"] not in {entry.get("path") for entry in identity.get("source_files", [])}:
        raise ValueError("Required execution source is not registered in the verified manifest")
    if canonical_digest({"source_files": identity["source_files"], "license": identity.get("license")}) != contract["source_digest"]:
        raise ValueError("Identity source manifest differs from reviewed source and license fingerprint")
    seen = set()
    for section in ("source_files", "weights"):
        for entry in identity[section]:
            path = resolve_asset_path(directory, entry["path"])
            if path in seen:
                raise ValueError("Duplicate identity asset paths")
            seen.add(path)
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Identity asset verification failed: {entry['path']}")
    return identity


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IdentityPredictor:
    def __init__(self, model_id, assets_root=DEFAULT_ASSETS, device="cuda"):
        aliases = {"adaface-ir100-webface12m": "adaface", "magface-iresnet100": "magface"}
        self.candidate_id = {value: key for key, value in aliases.items()}.get(model_id, model_id)
        model_id = aliases.get(model_id, model_id)
        self.model_id, self.device = model_id, torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if model_id == "sface":
            self.model = SFaceTorch(device=device)
            self.identity = {"official_url": "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
                             "license": "Apache-2.0", "weight_sha256": SFACE_SHA256}
            self.preprocessing = "RGB raw 0..255; graph applies (x-127.5)/128"
        elif model_id in ("adaface", "magface"):
            family = "AdaFace" if model_id == "adaface" else "MagFace"
            directory = resolve_asset_path(assets_root, family)
            self.identity = verified_identity(directory, expected_model=family)
            checkpoint = torch.load(resolve_asset_path(directory, self.identity["weights"][0]["path"]),
                                    map_location="cpu", weights_only=True)
            state = checkpoint["state_dict"]
            if model_id == "adaface":
                source = load_source(resolve_asset_path(directory, IDENTITY_CONTRACTS[family]["execution_source"]), "vision_adaface_official_net")
                self.model = source.build_model("ir_101")
                state = {key[6:]: val for key, val in state.items() if key.startswith("model.")}
                self.preprocessing = "BGR (x/255-0.5)/0.5; official IR_101 factory is depth 100"
            else:
                source = load_source(resolve_asset_path(directory, IDENTITY_CONTRACTS[family]["execution_source"]), "vision_magface_official_iresnet")
                self.model = source.iresnet100(pretrained=False)
                prefixes = ("module.features.", "features.module.", "features.")
                state = {key[len(prefix):]: val for key, val in state.items()
                         for prefix in prefixes if key.startswith(prefix)
                         and key[len(prefix):] in self.model.state_dict()}
                self.preprocessing = "BGR x/255; official gen_feat ToTensor without mean subtraction"
            self.model.load_state_dict(state, strict=True)
            del checkpoint, state
            self.model.to(device).eval()
        else:
            raise ValueError(f"Unknown identity model: {model_id}")

    @torch.inference_mode()
    def predict(self, aligned_rgb):
        rgb = validate_crop(aligned_rgb)
        if self.model_id == "sface":
            vector = self.model.embedding_bgr(rgb[:, :, ::-1])
        else:
            blob = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1).copy()).float()[None] / 255
            if self.model_id == "adaface":
                blob = (blob - .5) / .5
            output = self.model(blob.to(self.device))
            if isinstance(output, tuple):
                output = output[0]
            vector = output.cpu().numpy().reshape(-1)
        return {"model_id": self.candidate_id, "embedding": normalized(vector).tolist(),
                "dimensions": len(vector), "asset_identity": self.identity,
                "preprocessing": self.preprocessing, "alignment": "caller-shared-112x112",
                "runtime": "pytorch", "precision": "float32", "tta": False}
