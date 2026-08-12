import numpy as np
import pytest

from wbs_inference.geometry import paste_crop_mask, validate_bbox


def test_bbox_is_rounded_clipped_and_half_open():
    assert validate_bbox((-4.2, 1.2, 8.6, 12), (10, 8, 3)) == (0, 1, 8, 10)


def test_degenerate_bbox_is_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        validate_bbox((9, 2, 11, 5), (10, 8, 3))


def test_paste_restores_original_shape_and_zeroes_outside_bbox():
    result = paste_crop_mask(np.ones((2, 2), dtype=bool), (8, 9, 3), (2, 1, 6, 5))
    assert result.shape == (8, 9)
    assert result[1:5, 2:6].all()
    assert result.sum() == 16
