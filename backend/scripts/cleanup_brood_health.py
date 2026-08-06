from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    print(f"{'WOULD DELETE' if dry_run else 'DELETE'}: {path}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove generated Brood Health artifacts from older model versions."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = backend_root()
    targets = [
        root / "artifacts" / "models" / "brood_health",
        root / "artifacts" / "metrics" / "brood_health",
        root / "artifacts" / "reports" / "brood_health",
        root / "data" / "interim" / "brood_health_metrics.parquet",
        root / "data" / "interim" / "brood_health_metrics.meta.json",
    ]
    for target in targets:
        remove_path(target, dry_run=args.dry_run)

    if not args.dry_run:
        for directory in (
            root / "artifacts" / "models" / "brood_health",
            root / "artifacts" / "metrics" / "brood_health",
            root / "artifacts" / "reports" / "brood_health" / "eda",
            root / "artifacts" / "reports" / "brood_health" / "model",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        print("Brood Health generated artifacts were reset for version 4.0.")


if __name__ == "__main__":
    main()
