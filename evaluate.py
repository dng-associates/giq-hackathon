from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.model_evaluator import ModelEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoint.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default="DATASETS",
        help="Dataset directory or S3 prefix (e.g. s3://my-bucket/raw).",
    )
    parser.add_argument(
        "--filename",
        default="train.xlsx",
        help="Dataset filename to evaluate.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Chronological validation fraction.",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation.json",
        help="Where to save evaluation JSON.",
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.checkpoint)
    result = evaluator.evaluate_dataset(
        filename=args.filename,
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Evaluation saved to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
