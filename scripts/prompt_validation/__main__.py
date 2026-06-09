"""Prompt-validation harness — single entry point.

Run all suites (5 samples each), or target a suite / case:

    PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation
    PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation --list
    PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation \
        --suite collection_lifecycle
    PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation \
        --suite collection_lifecycle --case update-focus-swap --samples 10

Each model-driven case runs ``--samples`` times (default 5) to surface
variance; the ``retrieval`` suite is embedding-deterministic (1 pass).
"""
from __future__ import annotations

import argparse

from scripts.prompt_validation import (
    collection_lifecycle,
    novel_patterns,
    retrieval,
    skills_extractor,
)
from scripts.prompt_validation._harness import Harness, report

SUITES = {
    collection_lifecycle.NAME: collection_lifecycle,
    skills_extractor.NAME: skills_extractor,
    novel_patterns.NAME: novel_patterns,
    retrieval.NAME: retrieval,
}


def _list() -> None:
    for name, mod in SUITES.items():
        cases = [c[0] for c in getattr(mod, "CASES", [])]
        if name == "retrieval":
            from scripts.prompt_validation.fixtures import MESSAGES
            cases = [m.id for m in MESSAGES]
        print(f"\n{name}")
        for c in cases:
            print(f"  - {c}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="prompt_validation")
    ap.add_argument("--suite", choices=list(SUITES), help="run only this suite")
    ap.add_argument("--case", help="run only this case id within the suite")
    ap.add_argument("--samples", type=int, default=5, help="model runs per case (default 5)")
    ap.add_argument("--list", action="store_true", help="list suites and cases, then exit")
    args = ap.parse_args()

    if args.list:
        _list()
        return

    suites = [SUITES[args.suite]] if args.suite else list(SUITES.values())
    h = Harness()
    print(f"# Prompt validation — model={h.model}, samples={args.samples}")

    results = []
    for mod in suites:
        results.extend(mod.run(h, args.samples, args.case))

    n_fail = report(results, h.metrics)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
