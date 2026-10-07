from pathlib import Path

class FrameInfo(object):
    def __init__(self, filepath=None, landmarks_list=None, reviewed_masks=None, timing=None, geometry=None, source_sha256=None):
        self.filepath = filepath
        self.landmarks_list = landmarks_list or []
        self.motion_deg = 0
        self.motion_power = 0
        self.reviewed_masks = reviewed_masks or []
        self.timing = timing
        self.geometry = geometry or []
        self.source_sha256 = source_sha256
