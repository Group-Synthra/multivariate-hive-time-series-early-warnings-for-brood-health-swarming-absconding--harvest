from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    target_path = backend_root / "config/harvesting.yaml"
    section_path = (
        backend_root
        / "config/harvesting_probability_calibration_section.yaml"
    )

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target config: {target_path}")
    if not section_path.exists():
        raise FileNotFoundError(
            f"Missing calibration section: {section_path}"
        )

    target = yaml.safe_load(
        target_path.read_text(encoding="utf-8")
    )
    section = yaml.safe_load(
        section_path.read_text(encoding="utf-8")
    )
    if not isinstance(target, dict) or not isinstance(section, dict):
        raise TypeError("Both YAML files must contain mappings.")

    target.update(section)
    target_path.write_text(
        yaml.safe_dump(
            target,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print("Merged sections:", list(section))
    print("Updated:", target_path)


if __name__ == "__main__":
    main()
