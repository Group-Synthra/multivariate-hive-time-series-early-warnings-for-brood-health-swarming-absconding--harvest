from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.modules.absconding.lstm import LstmTrainingOptions, train_absconding_lstm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Absconding LSTM sequence comparison model."
    )
    parser.add_argument(
        "--config",
        default=str(BACKEND_ROOT / "config" / "absconding.yaml"),
    )
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-sequences", type=int, default=30_000)
    parser.add_argument("--max-validation-sequences", type=int, default=15_000)
    parser.add_argument("--max-test-sequences", type=int, default=15_000)
    args = parser.parse_args()

    result = train_absconding_lstm(
        backend_root=BACKEND_ROOT,
        config_path=args.config,
        options=LstmTrainingOptions(
            sequence_length=args.sequence_length,
            stride=args.stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            maximum_train_sequences=args.max_train_sequences,
            maximum_validation_sequences=args.max_validation_sequences,
            maximum_test_sequences=args.max_test_sequences,
        ),
    )
    validation = result["validation_metrics"]
    test = result["test_metrics"]
    print("Absconding LSTM training completed.")
    print(f"Validation PR-AUC: {validation['pr_auc']}")
    print(f"Validation recall: {validation['recall']}")
    print(f"Test PR-AUC: {test['pr_auc']}")
    print(f"Test recall: {test['recall']}")
    print("Model: artifacts/models/absconding/absconding_lstm_sequence.keras")
    print("Metrics: artifacts/metrics/absconding/lstm_comparison.json")


if __name__ == "__main__":
    main()
