"""Run read-only exploratory Phase 18D interpretation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.phase18d_interpretation import (  # noqa: E402
    phase18d_read_only_guard,
    run_phase18d,
    verify_official_artifacts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the official campaign and all artifacts without analysis",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from repository root")
    if args.verify_only:
        with phase18d_read_only_guard():
            verification = verify_official_artifacts()
        result = {
            key: value for key, value in verification.items() if key != "manifest"
        }
    else:
        summary = run_phase18d()
        result = {
            "phase": summary["phase"],
            "analysis_type": summary["analysis_type"],
            "source_classification": summary["source_classification"],
            "canonical_integrity": summary["canonical_integrity"],
            "artifacts": summary["artifacts"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
