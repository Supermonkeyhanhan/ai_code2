
from pathlib import Path

from engine import ENGINE_NAMES, evaluate_all, load_dataset, predict, train_all_models


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset.json"


def main() -> None:
    data = load_dataset(DATASET)
    assert len(data["intents"]) >= 30, "Expected at least 30 intents."
    assert sum(len(i.get("patterns", [])) for i in data["intents"]) >= 1000, "Dataset seems unexpectedly small."

    bundle = train_all_models(DATASET)
    results = evaluate_all(bundle)

    assert set(results) == set(ENGINE_NAMES)

    examples = [
        "Where is the library?",
        "How do I reset my university password?",
        "When is the Data Structures exam?",
        "How much is the Diploma in Computer Science?",
    ]
    for question in examples:
        for model in ENGINE_NAMES:
            intent, confidence, top = predict(bundle, model, question)
            assert isinstance(intent, str) and intent, f"Invalid intent for {model}"
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for {model}"
            assert top, f"No top predictions for {model}"

    print("Smoke tests passed.")
    for model in ENGINE_NAMES:
        r = results[model]
        print(
            f"{model}: Accuracy={r.accuracy:.4f}, "
            f"Precision={r.precision:.4f}, Recall={r.recall:.4f}, F1={r.f1:.4f}"
        )


if __name__ == "__main__":
    main()
