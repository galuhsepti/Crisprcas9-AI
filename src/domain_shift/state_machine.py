"""
Phase 14 experiment state machine.

Read-only audit pipeline:
    DESIGN -> FEASIBILITY -> ANALYSIS -> STATISTICAL_EVALUATION
    -> FREEZE -> FINAL_EXTERNAL_EVALUATION -> AUDIT -> COMPLETE

FINAL_EXTERNAL_EVALUATION is only reachable through FREEZE. Any attempt to
reach it earlier raises RuntimeError. An explicit external-read guard is
provided so that the locked external test file cannot be opened before the experiment
is frozen.
"""

from typing import List

_ORDER: List[str] = [
    "DESIGN",
    "FEASIBILITY",
    "ANALYSIS",
    "STATISTICAL_EVALUATION",
    "FREEZE",
    "FINAL_EXTERNAL_EVALUATION",
    "AUDIT",
    "COMPLETE",
]

_EXTERNAL_STATE = "FINAL_EXTERNAL_EVALUATION"
_FREEZE_STATE = "FREEZE"


class Phase14StateMachine:
    """Deterministic transition guard for the Phase 14 experiment."""

    def __init__(self, start: str = "DESIGN"):
        if start not in _ORDER:
            raise ValueError(f"Unknown state '{start}'")
        self.state = start
        self.frozen = False
        self.log: List[str] = [f"start:{start}"]

    @property
    def order(self) -> List[str]:
        return list(_ORDER)

    def transition(self, target: str) -> None:
        if target not in _ORDER:
            raise ValueError(f"Unknown target state '{target}'")
        current_idx = _ORDER.index(self.state)
        target_idx = _ORDER.index(target)
        if target_idx <= current_idx and target != self.state:
            raise RuntimeError(
                f"Illegal backward transition {self.state} -> {target}"
            )
        if target == _EXTERNAL_STATE:
            if current_idx < _ORDER.index(_FREEZE_STATE):
                raise RuntimeError(
                    "External evaluation is forbidden before the experiment is "
                    "frozen. Transition through FREEZE first."
                )
        if self.state == _FREEZE_STATE and target in ("DESIGN", "FEASIBILITY", "ANALYSIS", "STATISTICAL_EVALUATION", "FREEZE"):
            raise RuntimeError(
                "Experiment is frozen; design states may not be re-entered."
            )
        self.state = target
        self.log.append(f"transition:{target}")

    def freeze(self) -> None:
        self.transition("FREEZE")
        self.frozen = True
        self.log.append("frozen:True")

    def require(self, *states: str) -> None:
        if self.state not in states:
            raise RuntimeError(
                f"Current state '{self.state}' is not one of {states}"
            )

    def require_frozen(self) -> None:
        if not self.frozen:
            raise RuntimeError(
                "Operation requires a frozen experiment; external access "
                "before FREEZE is forbidden."
            )

    def assert_moreno_read_allowed(self, path: str = "") -> None:
        """
        Runtime guard: locked external-test content may only be read after FREEZE.
        os.stat-style existence checks are permitted earlier (D-7).
        """
        if not self.frozen:
            raise RuntimeError(
                f"Locked external-test content access pre-freeze is forbidden "
                f"(state={self.state}). {path}"
            )