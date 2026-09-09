#!/usr/bin/env python3
"""Run the Phase 16F evidence-only dataset-readiness gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_landscape.gates import build_phase16f_dataset_gate


def main() -> dict:
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    payload = build_phase16f_dataset_gate(timestamp=now.isoformat())
    path = Path("results") / f"phase16f_dataset_gate_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Phase 16F dataset gate complete: {path}")
    print("No modeling, Moreno access, pooling, transformation, or generalization claim performed")
    return payload


if __name__ == "__main__":
    main()