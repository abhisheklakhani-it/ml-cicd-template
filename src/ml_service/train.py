"""Command line entry point: `python -m ml_service.train --out models/model.joblib`.

This lives in its own module on purpose. If `model.py` itself were run with
`python -m`, pickle would record ModelBundle as `__main__.ModelBundle`, and the
API (whose `__main__` is uvicorn) could not load the file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_service.model import save, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model and save it.")
    parser.add_argument("--out", type=Path, default=Path("models/model.joblib"))
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Fail (exit 1) instead of saving if test accuracy is below this.",
    )
    args = parser.parse_args()
    bundle = train()
    if bundle.metrics["accuracy"] < args.min_accuracy:
        raise SystemExit(
            f"accuracy {bundle.metrics['accuracy']} is below --min-accuracy {args.min_accuracy}"
        )
    save(bundle, args.out)
    print(json.dumps({"saved": str(args.out), **bundle.metrics}))


if __name__ == "__main__":
    main()
