from __future__ import annotations

from pathlib import Path


def replace_once(
    text: str,
    old: str,
    new: str,
    *,
    description: str,
) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {description}; found {count}."
        )
    return text.replace(old, new, 1)


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]

    module_path = (
        backend_root
        / "src"
        / "multivari"
        / "modules"
        / "harvesting"
        / "research_model_comparison.py"
    )
    test_path = (
        backend_root
        / "tests"
        / "modules"
        / "harvesting"
        / "test_research_model_comparison.py"
    )

    module_text = module_path.read_text(encoding="utf-8")

    old_module = '''    if not math.isclose(weights.sum(), 1.0, rel_tol=1e-9):
        raise RuntimeError(
            "Session-balanced sample weights do not sum to one."
        )

    return weights
'''

    new_module = '''    # Preserve the relative class/session/event weighting while
    # keeping the average sample weight equal to one. Normalizing the
    # complete vector to sum to one makes regularization dominate
    # Logistic Regression and prevents XGBoost/LightGBM tree splits
    # because their weighted Hessian totals become extremely small.
    weights *= len(rows)

    if not math.isclose(weights.mean(), 1.0, rel_tol=1e-9):
        raise RuntimeError(
            "Session-balanced sample weights do not have mean one."
        )

    return weights
'''

    module_text = replace_once(
        module_text,
        old_module,
        new_module,
        description="old weight-normalization block",
    )
    module_path.write_text(module_text, encoding="utf-8")

    test_text = test_path.read_text(encoding="utf-8")
    test_text = replace_once(
        test_text,
        "    assert np.isclose(weights.sum(), 1.0)\n",
        "    assert np.isclose(weights.mean(), 1.0)\n",
        description="old weight-sum test assertion",
    )
    test_text = replace_once(
        test_text,
        '    assert np.isclose(positive_by_session["s1"], 0.25)\n',
        '    assert np.isclose(positive_by_session["s1"], 2.0)\n',
        description="old first session expected weight",
    )
    test_text = replace_once(
        test_text,
        '    assert np.isclose(positive_by_session["s2"], 0.25)\n',
        '    assert np.isclose(positive_by_session["s2"], 2.0)\n',
        description="old second session expected weight",
    )
    test_text = replace_once(
        test_text,
        "        0.5,\n",
        "        4.0,\n",
        description="old negative-class expected weight",
    )
    test_path.write_text(test_text, encoding="utf-8")

    print("Patched:", module_path)
    print("Patched:", test_path)
    print(
        "Session-balanced weights now retain mean 1.0 instead of "
        "summing to 1.0."
    )


if __name__ == "__main__":
    main()
