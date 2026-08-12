from dataclasses import replace

import cv2
import numpy as np

from wbs_inference.pipeline import WBSSegmenter


class FakePredictor:
    def set_image(self, image):
        self.image = np.asarray(image)

    def predict(self, *, point_coords, point_labels, box, mask_input, multimask_output, return_logits):
        height, width = self.image.shape[:2]
        positives = np.asarray(point_coords)[np.asarray(point_labels) == 1]
        center_x, center_y = positives.mean(axis=0)
        yy, xx = np.mgrid[:height, :width]
        radius = min(height, width) * 0.3
        logits = radius - np.hypot(xx - center_x, yy - center_y)
        low_res = cv2.resize(logits.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)[None]
        return logits[None].astype(np.float32), np.asarray([0.75], dtype=np.float32), low_res


def test_lightweight_pipeline_restores_bbox_mask_and_metadata(default_config):
    yy, xx = np.mgrid[:96, :128]
    image = np.stack(((xx * 2) % 255, (yy * 2) % 255, (xx + yy) % 255), axis=-1).astype(np.uint8)
    config = replace(
        default_config,
        preprocessing=replace(default_config.preprocessing, long_edge=96, num_superpixels=20),
        growing=replace(
            default_config.growing,
            max_iterations=0,
            refine_rounds=0,
        ),
        selection=replace(
            default_config.selection,
            retain_n=1,
            area_clusters=1,
        ),
    )
    result = WBSSegmenter(FakePredictor(), config, device="cpu").predict(image, (20, 12, 100, 84))
    assert result.mask.shape == image.shape[:2]
    assert result.mask[:12].sum() == 0
    assert result.mask[84:].sum() == 0
    assert result.mask[:, :20].sum() == 0
    assert result.mask[:, 100:].sum() == 0
    assert result.metadata["selection"]["retain_n"] == 1
    assert result.metadata["resolved_config"]["selection"]["retain_n"] == 1
