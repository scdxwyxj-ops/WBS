import json
from pathlib import Path

import pytest

from wbs_inference.config import HeuristicWeights, load_config, resolved_config_dict


def test_default_config_exposes_n_and_checkpoint_hash(default_config):
    assert default_config.selection.retain_n == 2
    assert default_config.model.checkpoint_sha256 == (
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
    )
    assert isinstance(default_config.selection.heuristic_weights, HeuristicWeights)


def test_resolved_config_omits_internal_hash(default_config):
    resolved = resolved_config_dict(default_config)
    assert "sha256" not in resolved


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload.update({"unknown": 1}), ValueError),
        (lambda payload: payload["selection"].pop("retain_n"), ValueError),
        (lambda payload: payload["selection"].update({"retain_n": "2"}), TypeError),
        (lambda payload: payload["preprocessing"].update({"slic_zero": False}), ValueError),
    ],
)
def test_config_is_strict(tmp_path, mutation, error):
    source = tmp_path / "source.json"
    repository_config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    payload = json.loads(repository_config.read_text(encoding="utf-8"))
    mutation(payload)
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(error):
        load_config(source)
