from dataclasses import replace

import numpy as np

from wbs_inference.preprocessing import preprocess


def test_preprocessing_is_deterministic_and_points_are_interior(default_config):
    yy, xx = np.mgrid[:80, :120]
    image = np.stack(((xx * 2) % 255, (yy * 3) % 255, ((xx + yy) * 4) % 255), axis=-1).astype(np.uint8)
    preprocessing = replace(default_config.preprocessing, long_edge=96, num_superpixels=18)
    first = preprocess(image, preprocessing)
    second = preprocess(image, preprocessing)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.segments, second.segments)
    for region in first.regions:
        x, y = np.rint(region.point_xy).astype(int)
        assert region.mask[y, x]
