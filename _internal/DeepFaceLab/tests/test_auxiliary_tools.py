"""Migration checks using real DFL metadata and distributed auxiliary weights."""

import copy
import importlib.util
import json
import pickle
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from core.interact import interact as io
from core.leras import nn
from core.leras.weight_io import load_weight_mapping
from core.imagelib import SegIEPolys, SegIEPolyType
from DFLIMG import DFLJPG
from ErrFaceFilter.ErrFaceFilter import LandmarkErrorClassifier, audit_directory, draw_directory
from facelib import FaceEnhancer, FANExtractor, S3FDExtractor, XSegNet
from mainscripts import Sorter, Util, Trainer, FacesetResizer, FacesetEnhancer, VideoEd, Merger, XSegUtil
from samplelib import PackedFaceset, SampleLoader
from tests.me_fixtures import make_aligned
from utils.label_face import label_face_filename
from utils.train_status_export import prepare_sample
from yaw_image_filter import evaluate_condition, select_directory

torch.set_num_threads(2)
BACKEND = Path(__file__).resolve().parents[1]
REPOSITORY = BACKEND.parents[1]


@pytest.fixture(autouse=True)
def cpu_helper_networks():
    nn.initialize(nn.DeviceConfig([]), data_format="NCHW")


def test_distributed_face_enhancer_loads_all_weights_and_runs():
    network = FaceEnhancer(place_model_on_cpu=True)
    weights = pickle.loads((BACKEND / "facelib" / "FaceEnhancer.npy").read_bytes())
    assert len(network.model.get_weights()) == len(weights) == 72
    np.testing.assert_array_equal(network.model.conv1.weight.detach().numpy(),
                                  weights["conv1/weight:0"].transpose(3, 2, 0, 1))
    result = network.enhance(np.zeros((64, 64, 3), dtype=np.float32))
    assert result.shape == (64, 64, 3) and np.isfinite(result).all()


def test_detector_and_landmark_distributed_weights_run():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    detector = S3FDExtractor(place_model_on_cpu=True)
    assert detector.extract(image) == []
    for landmarks_3d in (False, True):
        extractor = FANExtractor(landmarks_3D=landmarks_3d, place_model_on_cpu=True)
        points = extractor.extract(image, [(32, 32, 224, 224)])[0]
        assert points is not None and points.shape == (68, 2) and np.isfinite(points).all()


def test_auxiliary_weight_rejection_is_atomic():
    network = FaceEnhancer(place_model_on_cpu=True).model
    weights = pickle.loads((BACKEND / "facelib" / "FaceEnhancer.npy").read_bytes())
    original = network.conv1.weight.detach().clone()
    malformed = copy.copy(weights)
    malformed["conv1/weight:0"] = np.zeros_like(malformed["conv1/weight:0"])
    malformed.pop("out4x_conv1/bias:0")
    with pytest.raises(ValueError, match="names mismatch"):
        load_weight_mapping(network, malformed)
    assert torch.equal(network.conv1.weight, original)
    with pytest.raises(ValueError, match="names/count"):
        load_weight_mapping(network, {"param_0": np.zeros(tuple(original.shape), dtype=np.float32)})


def test_xseg_training_save_reload_and_missing_weights(tmp_path):
    network = XSegNet("XSeg", load_weights=False, training=True, weights_file_root=tmp_path,
                      place_model_on_cpu=True, optimizer=nn.RMSprop(lr=0.0001, name="opt"))
    torch.manual_seed(11)
    image = torch.rand(1, 3, 256, 256)
    target = torch.zeros(1, 1, 256, 256)
    target[:, :, 64:192, 64:192] = 1.0
    logits, prediction = network.flow(image)
    assert prediction.shape == target.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    network.opt.zero_grad()
    loss.backward()
    network.opt.step()
    assert np.isfinite(float(loss.detach()))
    with torch.no_grad():
        expected = network.flow(image)[1].detach().clone()
    network.save_weights()
    resumed = XSegNet("XSeg", load_weights=True, training=True, weights_file_root=tmp_path,
                      place_model_on_cpu=True, optimizer=nn.RMSprop(lr=0.0001, name="opt"))
    with torch.no_grad():
        assert torch.equal(resumed.flow(image)[1], expected)
    assert int(resumed.opt.iterations) == 1
    (tmp_path / "XSeg_256.pth").write_bytes(pickle.dumps({"param_0": np.zeros(1)}))
    with pytest.raises(Exception, match="加载失败"):
        XSegNet("XSeg", weights_file_root=tmp_path, place_model_on_cpu=True,
                raise_on_no_model_files=True)


@pytest.mark.parametrize("extension", ["pak", "zip"])
def test_faceset_real_metadata_pack_roundtrip(tmp_path, extension, monkeypatch):
    folder = make_aligned(tmp_path / "含 空格 的人脸", count=2)
    expected = {path.name: path.read_bytes() for path in folder.glob("*.jpg")}
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: False)
    archive = PackedFaceset.pack(folder, ext=extension, delete_original=False)
    packed = PackedFaceset.load(folder)
    assert len(packed) == 2
    for sample in packed:
        assert sample.read_raw_file() == expected[sample.filename]
        assert sample.load_bgr().shape == (128, 128, 3)
    for path in folder.glob("*.jpg"):
        path.unlink()
    PackedFaceset.unpack(folder)
    assert not archive.exists()
    assert {path.name: path.read_bytes() for path in folder.glob("*.jpg")} == expected


def test_person_zip_keeps_duplicate_basenames_separate(tmp_path, monkeypatch):
    folder = tmp_path / "people"
    make_aligned(folder / "person-A", count=1)
    make_aligned(folder / "person-B", count=1, offset=32)
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: True)
    PackedFaceset.pack(folder, ext="zip", delete_original=False)
    samples = PackedFaceset.load(folder)
    assert {sample.person_name for sample in samples} == {"person-A", "person-B"}
    assert samples[0].read_raw_file() != samples[1].read_raw_file()


def test_original_pak_absolute_final_offset_is_readable(tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / "aligned", count=2)
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: False)
    archive = PackedFaceset.pack(folder, delete_original=False)
    data = bytearray(archive.read_bytes())
    _, metadata_size = struct.unpack("<QQ", data[:16])
    final_offset_position = 16 + metadata_size + 8 * 2
    data[final_offset_position:final_offset_position + 8] = struct.pack("<Q", len(data))
    archive.write_bytes(data)
    assert all(sample.read_raw_file() == (folder / sample.filename).read_bytes()
               for sample in PackedFaceset.load(folder))


def test_absdiff_matches_exact_nearest_and_farthest_pixels(tmp_path, monkeypatch):
    for index, value in enumerate((0, 20, 200)):
        cv2.imwrite(str(tmp_path / f"{index}.png"), np.full((16, 16, 3), value, dtype=np.uint8))
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: True)
    similar, trash = Sorter.sort_by_absdiff(tmp_path)
    assert [Path(item[0]).name for item in similar] == ["0.png", "1.png", "2.png"] and trash == []
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: False)
    different, _ = Sorter.sort_by_absdiff(tmp_path)
    assert [Path(item[0]).name for item in different] == ["0.png", "2.png", "1.png"]


def test_pose_and_original_image_helpers_are_usable(tmp_path):
    folder = make_aligned(tmp_path / "aligned", count=2)
    selected = select_directory(folder, "abs(x) < 180 and abs(y) < 180")
    assert selected["selectedCount"] == 2
    with pytest.raises(ValueError):
        evaluate_condition("__import__('os').system('echo unsafe')", dict(x=0, y=0, r=0, ft=4))
    result = audit_directory(folder)
    assert len(result["samples"]) == 2 and result["invalidCount"] == 0
    assert draw_directory(folder)["count"] == 2
    samples = SampleLoader.load_face_samples([str(folder / "001.jpg")])
    image, full, priority = prepare_sample(samples[0], dict(eyes_prio=True, mouth_prio=True), 64, samples[0].face_type)
    assert image.shape == (64, 64, 3) and full.shape == priority.shape == (64, 64, 1)
    assert full.max() <= 1 and priority.max() > 0
    labeled = label_face_filename(image, "中文样本 001.jpg")
    assert labeled.shape == image.shape and np.isfinite(labeled).all()


def test_numpy_classifier_matches_original_xgboost_reference():
    reference = REPOSITORY / ".validation" / "auxiliary" / "cl-reference.npz"
    if not reference.exists():
        pytest.skip("Local original-runtime reference is not part of the source repository")
    data = np.load(reference)
    actual = LandmarkErrorClassifier().predict_proba(data["features"])
    np.testing.assert_allclose(actual, data["probability"], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actual > 0.5, data["probability"] > 0.5)


def test_web_asset_tool_uses_new_backend_metadata_and_sface(tmp_path):
    sys.path.insert(0, str(REPOSITORY / "webui" / "python"))
    from dfl_asset_tool import audit_directory as web_audit, build_pose_atlas
    folder = make_aligned(tmp_path / "aligned", count=2)
    audit = web_audit(folder)
    atlas = build_pose_atlas(folder)
    assert audit["total"] == atlas["total"] == 2
    from role_grouping import group_directory, MODEL_NAME
    groups = group_directory(folder, REPOSITORY / '_internal' / 'vision_models' / MODEL_NAME,
                             DFLJPG.load, lambda *args: None)
    assert groups['analyzedCount'] == 2 and groups['invalidCount'] == 0
    assert sum(group['memberCount'] for group in groups['groups']) == 2


def test_xseg_model_metadata_training_preview_and_resume(tmp_path, monkeypatch):
    import models
    source = make_aligned(tmp_path / "src", count=2)
    destination = make_aligned(tmp_path / "dst", count=2, offset=23)
    for directory in (source, destination):
        for path in directory.glob("*.jpg"):
            image = DFLJPG.load(path)
            polygons = SegIEPolys()
            polygon = polygons.add_poly(SegIEPolyType.INCLUDE)
            for x, y in [(20, 15), (105, 15), (105, 115), (20, 115)]:
                polygon.add_pt(x, y)
            image.set_seg_ie_polys(polygons)
            image.save()
    monkeypatch.setattr(io, "input_int", lambda *args, **kwargs: 2)
    monkeypatch.setattr(io, "input_str", lambda _message, default=None, *args, **kwargs: default or "XSeg")
    monkeypatch.setattr(io, "input_bool", lambda _message, default=False, **kwargs: default)
    monkeypatch.setattr(io, "input_in_time", lambda *args, **kwargs: False)
    monkeypatch.setattr(io, "input_skip_pending", lambda *args, **kwargs: None)
    folder = tmp_path / "model"
    folder.mkdir()
    model = models.import_model("XSeg")(is_training=True, debug=True, no_preview=True,
        saved_models_path=folder, training_data_src_path=source, training_data_dst_path=destination,
        cpu_only=True, silent_start=True)
    iteration, _ = model.train_one_iter()
    assert iteration == 1 and model.get_previews()
    model.save()
    metadata = pickle.loads((folder / "XSeg_data.dat").read_bytes())
    assert metadata["iter"] == 1 and metadata["options"]["face_type"] == "wf"
    assert (folder / "XSeg_256.pth").is_file() and (folder / "XSeg_256_opt.pth").is_file()
    model.finalize()
    resumed = models.import_model("XSeg")(is_training=False, debug=True, no_preview=True,
        saved_models_path=folder, cpu_only=True, silent_start=True)
    assert resumed.get_iter() == 1 and int(resumed.model.opt.iterations) == 1
    resumed.finalize()
    monkeypatch.setattr(nn.DeviceConfig, 'ask_choose_device', staticmethod(lambda **kwargs: nn.DeviceConfig.CPU()))
    # Model WF and fixture FULL differ, exercising aligned-landmark transforms;
    # these artificial fixtures intentionally contain no source-frame landmarks.
    XSegUtil.apply_xseg(source, folder)
    applied = DFLJPG.load(source / '000.jpg')
    assert applied.has_xseg_mask() and applied.get_xseg_mask().shape == (256, 256, 1)
    assert set(np.unique(applied.get_xseg_mask())) <= {0.0, 1.0}


def test_xseg_trainer_initialization_failure_exits_instead_of_hanging(tmp_path, monkeypatch):
    def missing_model(_name):
        raise ValueError("missing auxiliary model")
    monkeypatch.setattr(Trainer.models, "import_model", missing_model)
    with pytest.raises(RuntimeError, match="missing auxiliary model"):
        Trainer.main(model_class_name="XSeg", saved_models_path=tmp_path / "model",
            training_data_src_path=tmp_path / "src", training_data_dst_path=tmp_path / "dst",
            no_preview=True, cpu_only=True)


def test_metadata_restore_mask_export_and_resize_keep_dfl_data(tmp_path, monkeypatch):
    folder = make_aligned(tmp_path / "aligned", count=2)
    for path in folder.glob("*.jpg"):
        image = DFLJPG.load(path)
        image.set_xseg_mask(np.ones((128, 128, 1), dtype=np.float32))
        image.save()
    Util.save_faceset_metadata_folder(folder)
    before = DFLJPG.load(folder / "001.jpg").get_xseg_mask()
    for path in folder.glob("*.jpg"):
        pixels = cv2.imread(str(path))
        cv2.imwrite(str(path), pixels)
    assert not DFLJPG.load(folder / "001.jpg").has_data()
    Util.restore_faceset_metadata_folder(folder)
    np.testing.assert_array_equal(DFLJPG.load(folder / "001.jpg").get_xseg_mask(), before)
    Util.export_faceset_mask(folder)
    assert list(folder.glob("*_mask.jpg"))
    # The metadata-side mask export stays outside resize's aligned image set.
    for path in folder.glob("*_mask.jpg"):
        path.unlink()
    monkeypatch.setattr(io, "input_int", lambda *args, **kwargs: 256)
    monkeypatch.setattr(io, "input_str", lambda *args, **kwargs: "same")
    monkeypatch.setattr(io, "input_bool", lambda *args, **kwargs: False)
    FacesetResizer.process_folder(folder)
    resized = folder.parent / "aligned_resized" / "001.jpg"
    image = DFLJPG.load(resized)
    assert image.has_data() and image.get_shape()[:2] == (256, 256) and image.has_xseg_mask()


def test_video_encoding_extraction_and_denoise_keep_names_and_metadata(tmp_path):
    folder = make_aligned(tmp_path / 'aligned', count=2)
    names = sorted(path.name for path in folder.glob('*.jpg'))
    metadata = DFLJPG.load(folder / '001.jpg').get_dict()
    VideoEd.denoise_image_sequence(folder, factor=3)
    assert sorted(path.name for path in folder.glob('*.jpg')) == names
    assert DFLJPG.load(folder / '001.jpg').get_source_filename() == metadata['source_filename']
    video = tmp_path / 'new directory' / 'result.mp4'
    VideoEd.video_from_sequence(folder, video, ext='jpg', fps=8, lossless=True)
    assert video.is_file() and video.stat().st_size > 0
    frames = tmp_path / 'frames'
    VideoEd.extract_video(video, frames, output_ext='png', fps=0)
    assert len(list(frames.glob('*.png'))) == 2
    with pytest.raises(FileNotFoundError):
        VideoEd.extract_video(tmp_path / 'missing.mp4', tmp_path / 'missing output')


def test_guided_merger_configuration_is_complete_atomic_and_uses_learned_mask():
    import json
    from merger import MergerConfigMasked
    cfg = MergerConfigMasked()
    workers = Merger.apply_web_merge_config(cfg, json.dumps(dict(mode='seamless', workers=1,
        maskMode=4, erodeMask=-5, blurMask=20, motionBlur=2, faceScale=3, colorTransfer='rct',
        sharpenMode=2, sharpenAmount=30, superResolution=25, imageDenoise=15,
        bicubicDegrade=10, colorDegrade=8)))
    assert workers == 1 and cfg.mode == 'seamless' and cfg.mask_mode == 4
    assert cfg.erode_mask_modifier == -5 and cfg.blur_mask_modifier == 20
    assert cfg.color_transfer_mode == 1 and cfg.super_resolution_power == 25
    previous = vars(cfg).copy()
    with pytest.raises(ValueError, match='Invalid Web merge parameter'):
        Merger.apply_web_merge_config(cfg, '{"mode":"overlay","maskMode":4,"blurMask":999}')
    assert vars(cfg) == previous
    defaults = MergerConfigMasked()
    Merger.apply_web_merge_config(defaults, '{}')
    assert defaults.mask_mode == 4
