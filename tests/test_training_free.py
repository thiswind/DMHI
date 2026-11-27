"""Training-free contract of the dmhi package (IoTJ-R1, Reviewer 3).

The learned-metric network and the learned decoder were removed so that the
code matches the paper's zero-trained-parameters claim. These tests pin the
invariants the removal must preserve: the package never imports torch, the
constructor still accepts the deprecated keyword arguments of old
``config.json`` files, and checkpoints written by earlier releases still load.
"""
import pickle
import subprocess
import sys

import numpy as np

from dmhi.method.pipeline import RiemannianImputer


def test_import_does_not_pull_torch():
    # Run in a fresh interpreter with THIS repo's src/ first on the path
    # (a stale editable install of dmhi elsewhere on the machine would
    # otherwise be picked up by a bare `python -c`).
    import os
    import pathlib

    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = (
        "import sys, dmhi.method.pipeline as p; "
        "assert 'torch' not in sys.modules; "
        "assert p.__file__.startswith(sys.argv[1]), p.__file__; "
        "i = p.RiemannianImputer(d=12, use_learned_metric=True, epochs=5); "
        "assert i.ablation_no_metric is True and i._metric_net is None; "
        "print('OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, src],
        capture_output=True, text=True, check=True, env=env,
    )
    assert out.stdout.strip().endswith("OK")
    assert "torch" not in out.stderr


def test_source_tree_has_no_torch_import():
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[1] / "src" / "dmhi"
    offenders = [
        str(p) for p in pkg.rglob("*.py") if "import torch" in p.read_text()
    ]
    assert offenders == [], f"torch imports found in dmhi: {offenders}"


def test_deprecated_training_kwargs_are_accepted_and_ignored():
    # Keyword set of an old config.json that enabled the trainable branches.
    imp = RiemannianImputer(
        d=12,
        epochs=5,
        patience=3,
        batch_size=16,
        lr=1e-4,
        lambda_F=1e-5,
        lambda_smooth=1e-4,
        decoder_epochs=3,
        decoder_lr=1e-3,
        use_learned_metric=True,
        ablation_use_decoder=True,
        stage2_v2=True,
        stage2_realign=True,
    )
    assert imp.ablation_no_metric is True
    assert imp.use_learned_metric is False
    assert imp.ablation_use_decoder is False
    assert imp._metric_net is None
    assert imp._decoder is None
    assert not hasattr(imp, "_device")
    assert imp._needs_torch() is False


def test_deprecated_kwargs_do_not_change_default_output():
    X = np.load(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src/dmhi/utils/sample/X.npy"
    ).astype(np.float32)
    M = np.load(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src/dmhi/utils/sample/M.npy"
    ).astype(np.int8)
    n_train = max(8, int(round(len(X) * 0.7)))
    kw = dict(d=2, k=6, k_clle=5, embed_epochs=30)
    a = RiemannianImputer(**kw).build(X[:n_train], M[:n_train]).impute(X[n_train:], M[n_train:])
    b = RiemannianImputer(
        **kw, use_learned_metric=True, ablation_use_decoder=True, epochs=5
    ).build(X[:n_train], M[:n_train]).impute(X[n_train:], M[n_train:])
    assert np.array_equal(a, b)


def test_legacy_checkpoint_with_device_key_still_loads(tmp_path):
    X = np.load(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src/dmhi/utils/sample/X.npy"
    ).astype(np.float32)
    M = np.load(
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src/dmhi/utils/sample/M.npy"
    ).astype(np.int8)
    n_train = max(8, int(round(len(X) * 0.7)))
    imp = RiemannianImputer(d=2, k=6, k_clle=5, embed_epochs=30)
    imp.build(X[:n_train], M[:n_train])
    # Emulate a pickle written by the previous release: a `_device` entry
    # and a learned-metric flag are present in the instance dict.
    state = imp.__dict__.copy()
    state["_device"] = "cpu"
    state["use_learned_metric"] = True
    state["ablation_no_metric"] = False
    legacy = imp.__class__.__new__(imp.__class__)
    legacy.__dict__.update(state)
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(legacy, f)
    loaded = RiemannianImputer.load(str(path))
    assert loaded.ablation_no_metric is True
    assert loaded.use_learned_metric is False
    assert loaded._metric_net is None and loaded._decoder is None
    assert not hasattr(loaded, "_device")
    X_hat = loaded.impute(X[n_train:], M[n_train:])
    assert np.isfinite(X_hat).all()
