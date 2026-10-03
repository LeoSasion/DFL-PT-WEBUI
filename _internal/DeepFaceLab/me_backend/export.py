"""DFM-compatible ONNX export from the native ME checkpoint."""
import os
from pathlib import Path
import torch
from .engine import MEEngine


class DFMNetwork(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, face):
        swap, source_mask, destination_mask = self.network.predict(face.permute(0, 3, 1, 2))
        return destination_mask.permute(0, 2, 3, 1), swap.permute(0, 2, 3, 1), source_mask.permute(0, 2, 3, 1)


def export_dfm(model_directory, output=None):
    import onnx
    directory = Path(model_directory)
    engine = MEEngine.load(directory / 'me.pt', 'cpu')
    engine.network.eval()
    resolution = engine.config.resolution
    output = Path(output) if output else directory / 'model.dfm'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        torch.onnx.export(DFMNetwork(engine.network), torch.zeros(1, resolution, resolution, 3),
            str(temporary), input_names=['in_face:0'],
            output_names=['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0'],
            dynamic_axes={name: {0: 'batch'} for name in ('in_face:0', 'out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0')},
            opset_version=17, dynamo=False)
        model = onnx.load(str(temporary))
        onnx.checker.check_model(model)
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = 'model_type', 'ME-PyTorch'
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = 'face_type', engine.config.face_type
        onnx.save(model, str(temporary))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'[ME export] {output.resolve()}', flush=True)
    return output
