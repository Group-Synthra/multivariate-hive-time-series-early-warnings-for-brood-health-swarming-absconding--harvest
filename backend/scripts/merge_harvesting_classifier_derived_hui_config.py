from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    main_path = backend_root / "config/harvesting.yaml"
    section_path = (
        backend_root
        / "config/harvesting_classifier_derived_hui_section.yaml"
    )

    if not main_path.exists():
        raise FileNotFoundError(f"Missing main config: {main_path}")
    if not section_path.exists():
        raise FileNotFoundError(
            f"Missing classifier-derived HUI section: {section_path}"
        )

    main_config = yaml.safe_load(
        main_path.read_text(encoding="utf-8")
    ) or {}
    section_config = yaml.safe_load(
        section_path.read_text(encoding="utf-8")
    ) or {}

    if "classifier_derived_hui" not in section_config:
        raise ValueError(
            "Section file must define classifier_derived_hui."
        )

    main_config["classifier_derived_hui"] = section_config[
        "classifier_derived_hui"
    ]
    main_path.write_text(
        yaml.safe_dump(
            main_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print("Merged sections: ['classifier_derived_hui']")
    print(f"Updated: {main_path}")


if __name__ == "__main__":
    main()
