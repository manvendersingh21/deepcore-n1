"""CPU checks for displayed RGB-D evidence; no simulator or ideal object poses."""
import numpy as np
import pytest

from ui import sensor_overlay


def _camera():
    return np.array([[12., 0, 0], [0, 12., 0], [0, 0, 1]]), [0.25, 0.5, 1.8], np.diag([1., -1., -1.])


def test_backprojection_uses_optical_depth_and_camera_extrinsics():
    rgb = np.zeros((24, 24, 3), np.uint8)
    depth = np.zeros((24, 24), float)
    pixels = [(0, 0), (0, 12), (12, 0), (12, 12)]
    for (v, u), d, color in zip(pixels, [.8, .6, .4, 1.2],
                                [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        depth[v, u], rgb[v, u] = d, color

    packet = sensor_overlay.camera_packet(rgb, depth, *_camera())

    np.testing.assert_allclose(np.reshape(packet['points'], (-1, 3)),
                               [[.25, .5, 1.0], [.85, .5, 1.2], [.25, .1, 1.4], [1.45, -.7, .6]])
    np.testing.assert_array_equal(np.reshape(packet['colors'], (-1, 3)),
                                  [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(packet['frustum_corners'],
                               [[.25, .5, 1.1], [1.65, .5, 1.1], [1.65, -.9, 1.1], [.25, -.9, 1.1]])
    assert packet['provenance'] == 'actual_corrupted_rgbd'
    assert packet['image_size'] == [24, 24]
    assert packet['depth_image']
    assert packet['mask_image'] is None


@pytest.mark.parametrize('invalid', [0., np.nan, np.inf, -1., .05, 8.])
def test_invalid_depth_samples_are_absent_without_repair(invalid):
    rgb = np.full((24, 24, 3), 80, np.uint8)
    depth = np.full((24, 24), invalid)
    depth[0, 0] = .8
    before = depth.copy()

    packet = sensor_overlay.camera_packet(rgb, depth, *_camera())

    np.testing.assert_allclose(packet['points'], [.25, .5, 1.0])
    np.testing.assert_array_equal(depth, before)


def test_all_invalid_depth_produces_empty_cloud():
    rgb = np.zeros((24, 24, 3), np.uint8)
    depth = np.full((24, 24), np.nan)
    packet = sensor_overlay.camera_packet(rgb, depth, *_camera())
    assert packet['points'] == packet['colors'] == []
    assert np.isfinite(packet['frustum_corners']).all()
    assert 'No segmentation output supplied' in packet['mask_source']


@pytest.mark.parametrize('with_depth', [False, True])
def test_mask_provider_receives_exact_corrupted_inputs_once(with_depth, monkeypatch):
    rgb = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    depth = np.full((24, 24), .8, dtype=float)
    depth[0, 0], depth[1, 1] = np.nan, 0.
    original_rgb, original_depth = rgb.copy(), depth.copy()
    calls, encoded = [], []
    mask = np.zeros((24, 24), bool)
    mask[2:5, 3:7] = True

    class RGBOnly:
        def _masks(self, actual_rgb):
            calls.append((actual_rgb,))
            return {'core': mask}

    class RGBD:
        def _masks(self, actual_rgb, actual_depth):
            calls.append((actual_rgb, actual_depth))
            return {'core': mask}

    def capture_image(image):
        encoded.append(image.copy())
        return 'encoded-image'

    monkeypatch.setattr(sensor_overlay, '_jpeg', capture_image)
    packet = sensor_overlay.camera_packet(rgb, depth, *_camera(), RGBD() if with_depth else RGBOnly())

    assert len(calls) == 1
    assert calls[0][0] is rgb
    if with_depth:
        assert calls[0][1] is depth
        np.testing.assert_array_equal(calls[0][1], original_depth)
    assert len(calls[0]) == (2 if with_depth else 1)
    np.testing.assert_array_equal(rgb, original_rgb)
    np.testing.assert_array_equal(depth, original_depth)
    assert len(encoded) == 2
    np.testing.assert_array_equal(encoded[1][mask], np.tile([60, 235, 241], (mask.sum(), 1)))
    assert not encoded[1][~mask].any()
    assert packet['mask_image'] == 'encoded-image'
    assert 'actual corrupted input' in packet['mask_source']
