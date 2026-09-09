from __future__ import annotations

import json
from pathlib import Path

from engine import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    ENGINE_NAMES,
    build_quality_report,
    dataset_statistics,
    evaluate_all,
    evaluate_challenge_set,
    load_dataset,
    train_all_models,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset.json"
CHALLENGE = BASE_DIR / "challenge_test.json"


def main() -> None:
    data = load_dataset(DATASET)
    stats = dataset_statistics(data)
    quality = build_quality_report(data)

    print("Dataset statistics:")
    print(json.dumps(stats, indent=2))
    print("\nQuality report:")
    print(quality)

    assert quality.empty_patterns == 0, "Dataset contains empty training patterns."
    assert quality.conflicting_patterns == 0, "Dataset contains cross-intent pattern conflicts."

    models = train_all_models(DATASET)
    results = evaluate_all(models)
    print("\nHold-out evaluation:")
    for engine in ENGINE_NAMES:
        r = results[engine]
        print(
            f"{engine}: accuracy={r.accuracy:.3f}, precision={r.precision:.3f}, "
            f"recall={r.recall:.3f}, f1={r.f1:.3f}"
        )

    challenge_cases = json.loads(CHALLENGE.read_text(encoding="utf-8"))["cases"]
    print("\nChallenge-set evaluation:")
    for engine in ENGINE_NAMES:
        r = evaluate_challenge_set(
            models,
            challenge_cases,
            engine,
            DEFAULT_FALLBACK_THRESHOLD,
            DEFAULT_MARGIN_THRESHOLD,
        )
        print(
            f"{engine}: accuracy={r['accuracy']:.3f}, f1={r['f1']:.3f}, "
            f"fallbacks={r['fallback_count']}/{r['total']}"
        )

    print("\nSmoke test: PASS")


if __name__ == "__main__":
    main()
