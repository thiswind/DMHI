"""Every configs/*.json must parse and carry the required per-cell keys."""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "configs"
FILES = sorted(ROOT.glob("*.json"))

REQUIRED = {
    "dataset",
    "missing_rate",
    "pattern",
    "block_size",
    "seed",
    "d",
    "k",
    "k_clle",
    "epochs",
    "lr",
    "batch_size",
    "patience",
    "run_id",
}


def test_inventory():
    assert len(FILES) == 55


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_required_keys(path):
    cfg = json.loads(path.read_text())
    assert REQUIRED <= cfg.keys(), f"missing: {sorted(REQUIRED - cfg.keys())}"
    assert cfg["dataset"] in path.name
    assert int(path.stem.rsplit("seed", 1)[1]) == cfg["seed"]
    assert 0.0 < cfg["missing_rate"] <= 1.0
    assert cfg["d"] >= 1
    assert cfg["k"] >= 1
