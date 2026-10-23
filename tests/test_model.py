from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import pytest

from ml_service.model import ModelBundle, load, load_data, train
from ml_service.train import main


def test_quality_gate(bundle: ModelBundle) -> None:
    # A model that gets worse than this must not ship.
    assert bundle.metrics["accuracy"] >= 0.9
    assert bundle.metrics["macro_f1"] >= 0.9


def test_training_is_reproducible(bundle: ModelBundle) -> None:
    assert train().metrics == bundle.metrics


def test_predict_shape_and_probabilities(bundle: ModelBundle) -> None:
    x, y, _, names = load_data()
    preds = bundle.predict(x[:5].tolist())
    assert len(preds) == 5
    for p in preds:
        probs = p["probabilities"]
        assert isinstance(probs, dict)
        assert set(probs) == set(names)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
        assert p["label"] in names


def test_save_load_roundtrip(model_path: Path, bundle: ModelBundle) -> None:
    loaded = load(model_path)
    assert loaded.metrics == bundle.metrics
    assert loaded.feature_names == bundle.feature_names


def test_load_rejects_wrong_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(TypeError):
        load(path)


def test_cli_writes_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "m.joblib"
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out)])
    main()
    assert load(out).metrics["accuracy"] >= 0.9


def test_cli_quality_gate_blocks_bad_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "m.joblib"
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out), "--min-accuracy", "1.01"])
    with pytest.raises(SystemExit, match="below --min-accuracy"):
        main()
    assert not out.exists()


def test_model_from_cli_loads_in_another_process(tmp_path: Path) -> None:
    # Train exactly as the Dockerfile does, then load in this process.
    out = tmp_path / "m.joblib"
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ml_service.train", "--out", str(out)], check=True
    )
    assert load(out).metrics["accuracy"] >= 0.9
