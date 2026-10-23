"""Train, save and load the model.

The wine dataset ships with scikit-learn, so training needs no download.
Swap `load_data` and `build_pipeline` for your own model; the API and the
CI/CD pipeline stay the same.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


@dataclass
class ModelBundle:
    """Everything the API needs: the fitted pipeline plus its metadata."""

    pipeline: Pipeline
    feature_names: list[str]
    class_names: list[str]
    metrics: dict[str, float]

    def predict(self, rows: list[list[float]]) -> list[dict[str, object]]:
        x = np.asarray(rows, dtype=float)
        probs = self.pipeline.predict_proba(x)
        results: list[dict[str, object]] = []
        for p in probs:
            best = int(np.argmax(p))
            results.append(
                {
                    "label": self.class_names[best],
                    "confidence": round(float(p[best]), 4),
                    "probabilities": {
                        name: round(float(v), 4) for name, v in zip(self.class_names, p, strict=True)
                    },
                }
            )
        return results


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    data = load_wine()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def train() -> ModelBundle:
    x, y, feature_names, class_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED
    )
    pipeline = build_pipeline().fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "test_size": float(len(y_test)),
    }
    return ModelBundle(pipeline, feature_names, class_names, metrics)


def save(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path: Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model and save it.")
    parser.add_argument("--out", type=Path, default=Path("models/model.joblib"))
    args = parser.parse_args()
    bundle = train()
    save(bundle, args.out)
    print(json.dumps({"saved": str(args.out), **bundle.metrics}))


if __name__ == "__main__":
    main()
