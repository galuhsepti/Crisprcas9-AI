"""Guarded Phase 18C runner; defaults to verification with zero training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_protocols.phase18c_access import (  # noqa: E402
    phase18c_access_guard,
)


def run_preflight():
    """Import execution dependencies only inside the campaign-wide guard."""
    from src.multidomain.campaign import run_preflight as guarded_preflight

    return guarded_preflight()


def run_campaign(protocol_confirmation, implementation_confirmation):
    """Import the training path only inside the campaign-wide guard."""
    from src.multidomain.campaign import run_campaign as guarded_campaign

    return guarded_campaign(protocol_confirmation, implementation_confirmation)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify locks without fitting (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="enable the training path after digest confirmation",
    )
    parser.add_argument(
        "--confirm-protocol",
        help="full approved protocol SHA-256; required with --execute",
    )
    parser.add_argument(
        "--confirm-implementation",
        help="exact implementation SHA-256 reported by preflight",
    )
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
        run_dir = run_campaign(
            args.confirm_protocol, args.confirm_implementation
        )
        relative = run_dir.relative_to(ROOT).as_posix()
        print(f"Phase 18C campaign directory: {relative}")
        return run_dir


if __name__ == "__main__":
    main()
