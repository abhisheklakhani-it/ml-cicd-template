from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_service import app as app_module
from ml_service.model import ModelBundle, save, train


@pytest.fixture(scope="session")
def bundle() -> ModelBundle:
    return train()


@pytest.fixture(scope="session")
def model_path(bundle: ModelBundle, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    save(bundle, path)
    return path


@pytest.fixture()
def client(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(app_module, "MODEL_PATH", model_path)
    with TestClient(app_module.app) as c:
        yield c
