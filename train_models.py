
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import ENGINE_NAMES, evaluate_all, train_all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate University Chatbot models.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output", default="evaluation_summary.json")
    args = parser.parse_args()

    models = train_all_models(args.dataset)
    results = evaluate_all(models)

    summary = {}
    for engine in ENGINE_NAMES:
        result = results[engine]
        summary[engine] = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "macro_f1": result.macro_f1,
            "training_seconds": result.training_seconds,
        }

    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
