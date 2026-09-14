"""Generate a Phase 18B protocol only; never execute the Phase 18C matrix."""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_protocols.multidomain_protocol import (  # noqa: E402
    audit_only_guard,
    build_protocol,
    lock_protocol,
    serialize,
)


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(
            "Run from repository root (existing loader convention)"
        )
    with audit_only_guard():
        payload = lock_protocol(build_protocol())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = ROOT / "results" / f"phase18b_multidomain_protocol_{stamp}.json"
        # Exclusive creation prevents replacing an earlier preregistration.
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialize(payload))
        print(f"Phase 18B decision: {payload['final_decision']}")
        print(f"Artifact: {path.relative_to(ROOT).as_posix()}")
        print(f"Protocol SHA-256: {payload['protocol_sha256']}")
        print(f"Split counts: {payload['split_counts']}")
        print(
            f"Planned runs (not executed): {payload['budget']['total_runs']}"
        )
    return payload


if __name__ == "__main__":
    main()
