"""Generate the Phase 18G protocol only; never execute Phase 18H."""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_protocols.phase18g_positional_region_protocol import (  # noqa
    build_protocol,
    lock_protocol,
    phase18g_protocol_guard,
    serialize,
)


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from repository root")
    with phase18g_protocol_guard():
        payload = lock_protocol(build_protocol())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = (
            ROOT
            / "results"
            / (f"phase18g_positional_region_protocol_{stamp}.json")
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialize(payload))
    print(f"Phase 18G decision: {payload['final_decision']}")
    print(f"Artifact: {path.relative_to(ROOT).as_posix()}")
    print(f"Protocol SHA-256: {payload['protocol_sha256']}")
    print(
        "Planned Phase 18H new fits (not executed): "
        f"{payload['budget']['required_new_fits']}"
    )
    return payload


if __name__ == "__main__":
    main()
