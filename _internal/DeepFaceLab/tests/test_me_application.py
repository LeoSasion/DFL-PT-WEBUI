"""Native ME model application through the retained merger and DFM contracts."""
import cv2
import numpy as np
import onnxruntime as ort
import torch

from facelib import LandmarksProcessor
from me_backend.config import MEConfig
from me_backend.engine import MEEngine
from me_backend.export import export_dfm
from me_backend.model_adapter import MEInferenceModel
from merger import FrameInfo
from merger.MergeMasked import MergeMasked

torch.set_num_threads(2)


def create_model(directory):
    engine = MEEngine(MEConfig(resolution=64, ae_dims=32, e_dims=16,
        d_dims=16, d_mask_dims=16, batch_size=1), 'cpu')
    engine.save(directory / 'me.pt')
    return engine


def test_dfm_real_onnx_inference_matches_me(tmp_path):
    engine = create_model(tmp_path)
    output = export_dfm(tmp_path)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    assert [value.name for value in session.get_inputs()] == ['in_face:0']
    assert [value.name for value in session.get_outputs()] == [
        'out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0']
    image = np.random.default_rng(4).random((2, 64, 64, 3), dtype=np.float32)
    destination_mask, swap, source_mask = session.run(None, {'in_face:0': image})
    expected_swap, expected_source, expected_destination = engine.predict(image.transpose(0, 3, 1, 2))
    for actual, expected in zip((swap, source_mask, destination_mask),
        (expected_swap, expected_source, expected_destination)):
        np.testing.assert_allclose(actual, expected.transpose(0, 2, 3, 1), rtol=2e-4, atol=2e-6)


def test_existing_merger_accepts_native_me_checkpoint(tmp_path):
    directory = tmp_path / 'model' / 'sample'
    create_model(directory)
    model = MEInferenceModel(saved_models_path=directory.parent, force_model_name='sample', cpu_only=True)
    predictor, shape, config = model.get_MergerConfig()
    config.color_transfer_mode = 0
    config.mask_mode = 4
    yy, xx = np.mgrid[:128, :128]
    image = np.stack((xx * 2, yy * 2, (xx + yy)), axis=-1).astype(np.uint8)
    frame = tmp_path / 'frame.png'
    ok, buffer = cv2.imencode('.png', image)
    assert ok
    buffer.tofile(frame)
    info = FrameInfo(filepath=frame, landmarks_list=[LandmarksProcessor.get_canonical_68(128)])
    result = MergeMasked(predictor, shape, None, None, config, info)
    assert result.shape == (128, 128, 4)
    assert np.isfinite(result).all() and result[..., 3].max() > 0
    assert np.any(result[..., :3] != image)
