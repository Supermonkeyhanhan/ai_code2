from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import ENGINE_NAMES, average_inference_ms, evaluate_all, load_dataset, train_all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate University Chatbot models.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output", default="evaluation_summary.json")
    args = parser.parse_args()

    models = train_all_models(args.dataset)
    results = evaluate_all(models)
    data = load_dataset(args.dataset)

    summary = {
        "dataset": {
            "intents": len(data["intents"]),
            "patterns": sum(len(item.get("patterns", [])) for item in data["intents"]),
            "train_samples": len(models.x_train),
            "test_samples": len(models.x_test),
            "split": "80/20 stratified hold-out",
        },
        "models": {},
    }

    for engine in ENGINE_NAMES:
        result = results[engine]
        summary["models"][engine] = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "macro_precision": result.macro_precision,
            "macro_recall": result.macro_recall,
            "macro_f1": result.macro_f1,
            "training_seconds": result.training_seconds,
            "average_inference_ms": average_inference_ms(models, engine, models.x_test, repeats=3),
        }

    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
