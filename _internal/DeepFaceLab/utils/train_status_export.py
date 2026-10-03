"""Sample preparation helpers for status/export tooling, independent of TF."""

import cv2
import numpy as np

from core import imagelib
from facelib import FaceType, LandmarksProcessor


def get_full_face_mask(sample, sample_bgr, sample_landmarks):
    mask = sample.get_xseg_mask()
    if mask is None:
        mask = LandmarksProcessor.get_image_hull_mask(sample_bgr.shape, sample_landmarks,
                                                     eyebrows_expand_mod=sample.eyebrows_expand_mod)
    elif mask.shape[:2] != sample_bgr.shape[:2]:
        mask = cv2.resize(mask, sample_bgr.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    return np.clip(imagelib.normalize_channels(mask, 1), 0.0, 1.0)


def get_eyes_mask(sample_bgr, sample_landmarks):
    mask = np.clip(LandmarksProcessor.get_image_eye_mask(sample_bgr.shape, sample_landmarks), 0.0, 1.0)
    return np.where(mask > 0.1, mask + 1.0, mask)


def get_mouth_mask(sample_bgr, sample_landmarks):
    mask = np.clip(LandmarksProcessor.get_image_mouth_mask(sample_bgr.shape, sample_landmarks), 0.0, 1.0)
    return np.where(mask > 0.1, mask + 2.0, mask)


def get_eyes_mouth_mask(sample_bgr, sample_landmarks):
    eyes = LandmarksProcessor.get_image_eye_mask(sample_bgr.shape, sample_landmarks)
    mouth = LandmarksProcessor.get_image_mouth_mask(sample_bgr.shape, sample_landmarks)
    return np.clip(eyes + mouth, 0.0, 1.0)


def get_full_face_eyes(sample, sample_bgr, sample_landmarks):
    full = get_full_face_mask(sample, sample_bgr, sample_landmarks)
    eyes = get_eyes_mask(sample_bgr, sample_landmarks) * (full != 0.0)
    mouth = get_mouth_mask(sample_bgr, sample_landmarks) * (full != 0.0)
    encoded = np.where(eyes > 1.0, eyes, full)
    return np.where(mouth > 2.0, mouth, encoded)


def get_masks(sample, sample_bgr, sample_landmarks, eye_prio=False, mouth_prio=False):
    full = get_full_face_mask(sample, sample_bgr, sample_landmarks)
    priority = np.zeros_like(full)
    if eye_prio:
        priority += LandmarksProcessor.get_image_eye_mask(sample_bgr.shape, sample_landmarks)
    if mouth_prio:
        priority += LandmarksProcessor.get_image_mouth_mask(sample_bgr.shape, sample_landmarks)
    return full, np.clip(priority, 0.0, 1.0) * (full != 0.0)


def get_input_image(image, sample_face_type, sample_landmarks, resolution, face_type):
    if face_type != sample_face_type:
        matrix = LandmarksProcessor.get_transform_mat(sample_landmarks, resolution, face_type)
        image = cv2.warpAffine(image, matrix, (resolution, resolution),
                              borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR)
    elif image.shape[:2] != (resolution, resolution):
        image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return imagelib.normalize_channels(image, 1) if image.ndim == 2 else image


def prepare_sample(sample, options, resolution, face_type_enum):
    image = sample.load_bgr()
    full, priority = get_masks(sample, image, sample.landmarks,
                              options.get("eyes_prio", False), options.get("mouth_prio", False))
    return tuple(get_input_image(item, sample.face_type, sample.landmarks, resolution, face_type_enum)
                 for item in (image, full, priority))


def data_format_change(image):
    if image.ndim == 2:
        image = image[..., None]
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])


def print_sample_status(sample):
    image = sample.load_bgr() if hasattr(sample, "load_bgr") else np.asarray(sample)
    print(dict(shape=list(image.shape), maximum=float(np.max(image)), minimum=float(np.min(image))))
