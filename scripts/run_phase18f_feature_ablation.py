"""Preflight or execute the locked Phase 18F feature-ablation campaign."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_protocols.phase18c_access import (  # noqa: E402
    phase18c_access_guard,
)
from src.feature_ablation.phase18f import (  # noqa: E402
    run_campaign,
    run_preflight,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-protocol")
    parser.add_argument("--confirm-implementation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from repository root")
    with phase18c_access_guard():
        preflight = run_preflight()
        print(json.dumps(preflight, indent=2, sort_keys=True))
        if not args.execute:
            return preflight
        results_dir, models_dir = run_campaign(
            args.confirm_protocol, args.confirm_implementation
        )
        print(f"Phase 18F results: {results_dir.relative_to(ROOT).as_posix()}")
        print(f"Phase 18F models: {models_dir.relative_to(ROOT).as_posix()}")
        return results_dir, models_dir


if __name__ == "__main__":
    main()
