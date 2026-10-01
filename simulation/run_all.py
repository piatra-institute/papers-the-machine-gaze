"""Reproduce every number and figure.

    cd simulation && uv run run_all.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from analyses import run

OUT = Path(__file__).parent / "output"


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = run()
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    from figures import make_all
    make_all(results, OUT / "figures")
    failed = [k for k, v in results["checks"].items() if not v]
    print(f"checks: {len(results['checks']) - len(failed)}/{len(results['checks'])} in {time.time() - t0:.0f} s")
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
