"""Shipped processed datasets must be structurally valid.

Asserts the 17 redistributed domains load, have consistent (N, T, D) shapes
between X and M arrays, and masks are binary. Uses mmap so the whole suite
runs in seconds.
"""
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS = ("train", "val", "test")
DOMAINS = sorted(d.name for d in ROOT.iterdir() if d.is_dir())

NOT_IN_REPO = {"physionet2012", "mimic", "electricity"}


def test_seventeen_domains_shipped():
    assert len(DOMAINS) == 17, DOMAINS
    assert not (set(DOMAINS) & NOT_IN_REPO)


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_shapes_and_masks(domain):
    d = ROOT / domain
    shapes = {}
    for split in SPLITS:
        X = np.load(d / f"X_{split}.npy", mmap_mode="r")
        M = np.load(d / f"M_{split}.npy", mmap_mode="r")
        assert X.ndim == 3
        assert M.shape == X.shape
        assert str(M.dtype) == "int8"
        shapes[split] = X.shape
    assert len({s[1:] for s in shapes.values()}) == 1, "T/D must match across splits"
    assert shapes["train"][0] > shapes["val"][0]
