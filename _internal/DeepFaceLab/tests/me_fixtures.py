"""Artificial aligned DFLJPGs: test the file pipeline, not face-swap quality."""
from pathlib import Path
import cv2
import numpy as np
from DFLIMG import DFLJPG
from facelib import LandmarksProcessor


def make_aligned(directory, count=4, offset=0):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    size = 128
    landmarks = LandmarksProcessor.get_canonical_68(size)
    yy,xx = np.mgrid[:size,:size]
    for i in range(count):
        image = np.stack(((xx+i*7+offset)%128/127,(yy+i*3)%128/127,
                          .35+.2*np.sin((xx+yy+offset)/18)),axis=-1)
        path = directory/f'{i:03d}.jpg'
        ok, buf = cv2.imencode('.jpg',(image*255).astype(np.uint8))
        assert ok
        buf.tofile(path)
        dfl = DFLJPG.load(str(path))
        dfl.set_face_type('full_face')
        dfl.set_landmarks(landmarks.tolist())
        dfl.set_source_filename('synthetic-test-frame.png')
        if i%2:
            mask = (((xx-64)/48)**2+((yy-64)/57)**2<1).astype(np.float32)[...,None]
            dfl.set_xseg_mask(mask)
        dfl.save()
    return directory
