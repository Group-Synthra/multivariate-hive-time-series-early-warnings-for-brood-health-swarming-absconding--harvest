from __future__ import annotations

import json
from pathlib import Path

from multivari.modules.harvesting.classifier_derived_hui import (
    build_classifier_derived_hui_dataset_from_config,
)


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    result = build_classifier_derived_hui_dataset_from_config(
        backend_root=backend_root,
        config_path="config/harvesting.yaml",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
