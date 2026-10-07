"""Bounded real PySide6 XSeg editor open/draw/save acceptance on private fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_internal/DeepFaceLab'))


def acceptance(destination):
    import cv2
    import numpy as np
    from PySide6.QtTest import QTest
    from PySide6.QtCore import QPoint, Qt
    from core.qtex import QImage_from_np, QImage_to_np
    from core.qtex.qt_compat import QApplication, QT_BINDING
    from DFLIMG import DFLIMG
    from XSegEditor.XSegEditor import MainWindow, LoaderQSubprocessor, QUIConfig
    from XSegEditor.QStringDB import QStringDB
    from XSegEditor.QIconDB import QIconDB
    from XSegEditor.QCursorDB import QCursorDB
    from XSegEditor.QImageDB import QImageDB
    assert QT_BINDING == 'PySide6'
    destination.mkdir(parents=True, exist_ok=False)
    images = destination / 'aligned'
    config = destination / 'config'
    images.mkdir(); config.mkdir()
    image = np.zeros((194, 259, 3), np.uint8)
    image[:, :, 0] = np.arange(259, dtype=np.uint16).astype(np.uint8)
    image[:, :, 1] = np.arange(194, dtype=np.uint8)[:, None]
    app = QApplication.instance() or QApplication([])
    # Width 259 exercises the real Qt 4-byte row padding path.
    assert np.array_equal(QImage_to_np(QImage_from_np(image)), image)
    target = images / 'fixture.jpg'
    cv2.imwrite(str(target), image)
    dfl = DFLIMG.load(target)
    dfl.set_face_type('whole_face')
    dfl.set_landmarks(np.column_stack((np.linspace(45, 215, 68), np.linspace(40, 155, 68))).astype(np.float32))
    dfl.save()
    pixels_before = cv2.imread(str(target))
    old_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assets = ROOT / '_internal/DeepFaceLab/XSegEditor/gfx'
    QUIConfig.initialize(); QStringDB.initialize()
    QIconDB.initialize(assets / 'icons'); QCursorDB.initialize(assets / 'cursors'); QImageDB.initialize(assets / 'images')
    # One genuine loader worker is sufficient for one image; no mocked loader.
    LoaderQSubprocessor.process_info_generator = lambda self: iter([('QA-CPU', {}, {})])
    window = MainWindow(images, config)
    window.resize(1000, 750); window.show()
    deadline = time.monotonic() + 20
    while window.loading_frame is not None and time.monotonic() < deadline:
        app.processEvents(); QTest.qWait(20)
    if window.loading_frame is not None:
        raise TimeoutError('XSeg editor loader did not finish within 20 seconds')
    operator = window.canvas.op
    app.processEvents()
    points = [(75, 55), (185, 55), (185, 140), (75, 140), (75, 55)]
    for point in points:
        position = operator.img_to_cli_pt(np.array(point, np.float32))
        QTest.mouseMove(operator, QPoint(*position.astype(int)))
        QTest.mouseClick(operator, Qt.LeftButton, pos=QPoint(*position.astype(int)))
        app.processEvents()
    polys = operator.get_ie_polys()
    assert polys.has_polys() and polys.get_pts_count() == 4
    assert window.grab().save(str(destination / 'editor.png'))
    window.canvas_finalize(target)
    loaded = DFLIMG.load(target)
    assert loaded.get_seg_ie_polys().get_pts_count() == 4
    assert np.array_equal(cv2.imread(str(target)), pixels_before)
    saved_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert saved_digest != old_digest
    window.close(); app.processEvents()
    result = {'ok': True, 'binding': QT_BINDING, 'realLoaderWorker': True, 'realEditorWindow': True,
              'mousePolygonPoints': 4, 'savedAndReloaded': True, 'jpegPixelsPreserved': True,
              'oddWidthQImageRoundtrip': True, 'originalSha256': old_digest, 'savedSha256': saved_digest,
              'platform': os.environ.get('QT_QPA_PLATFORM', 'native')}
    (destination / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offscreen', action='store_true')
    args = parser.parse_args()
    if args.offscreen:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    acceptance(args.output.resolve())
