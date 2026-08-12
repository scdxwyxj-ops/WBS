from dataclasses import replace

import numpy as np

from wbs_inference.config import HeuristicWeights
from wbs_inference.selection import Proposal, select_proposal


def _proposal(rows: int, score: float, iteration: int) -> Proposal:
    mask = np.zeros((20, 20), dtype=bool)
    mask[:rows] = True
    return Proposal(mask, score, None, None, iteration, None, None)


def test_default_retain_n_is_applied(default_config):
    image = np.tile(np.arange(20, dtype=np.uint8)[None, :, None], (20, 1, 3))
    proposals = [_proposal(2, 0.1, 1), _proposal(8, 0.7, 2), _proposal(16, 0.9, 3)]
    selection = replace(
        default_config.selection,
        target_area_ratio=0.4,
        heuristic_weights=HeuristicWeights(1.0, 0.0, 0.0, 0.0, 0.0),
    )
    selected, metadata = select_proposal(image, proposals, selection)
    assert metadata["retain_n"] == 2
    assert metadata["retained_pool_size"] == 2
    assert selected.iteration == 2


def test_retain_n_can_be_overridden(default_config):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    proposals = [_proposal(2, 0.2, 1), _proposal(8, 0.9, 2), _proposal(16, 0.4, 3)]
    selection = replace(default_config.selection, retain_n=3)
    selected, metadata = select_proposal(image, proposals, selection)
    assert metadata["retain_n"] == 3
    assert metadata["retained_pool_size"] == 3
    assert selected.iteration == 2
    assert selected.sam_score == 0.9
