"""Exact Phase 18C shared encoder with explicit domain-specific heads."""

from __future__ import annotations

import random
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.experiment_protocols.multidomain_protocol import DOMAINS
from src.multidomain.data import TrainingStatistics


def set_deterministic_runtime(seed: int) -> None:
    """Set the locked CPU random and deterministic-computation contract."""
    if seed not in range(2**32):
        raise ValueError("Seed must be a uint32")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


class SequenceEncoder(nn.Module):
    """Canonical parallel-convolution encoder without its output layer."""

    def __init__(self):
        super().__init__()
        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, 64, kernel_size=k, padding=k // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for k in (5, 7, 9)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(192, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("CNN input must have three dimensions")
        if inputs.shape[1:] == (30, 4):
            inputs = inputs.permute(0, 2, 1)
        if inputs.shape[1:] != (4, 30):
            raise ValueError("CNN input must have shape (n,30,4) or (n,4,30)")
        encoded = [
            self.global_pool(branch(inputs)).squeeze(-1)
            for branch in self.conv_branches
        ]
        return self.dropout(self.relu(self.dense(torch.cat(encoded, dim=1))))


class DomainAwareCNN(nn.Module):
    """Shared encoder and raw-unit affine outputs routed by observed domain."""

    def __init__(
        self,
        statistics: Mapping[str, TrainingStatistics],
        run_seed: int,
    ):
        super().__init__()
        unknown = set(statistics) - set(DOMAINS)
        if not statistics or unknown:
            raise ValueError("Statistics must contain registered domains only")
        set_deterministic_runtime(run_seed)
        self.encoder = SequenceEncoder()
        self.heads = nn.ModuleDict()
        self._domain_order = tuple(
            domain for domain in DOMAINS if domain in statistics
        )
        for domain in self._domain_order:
            domain_index = DOMAINS.index(domain)
            torch.manual_seed(run_seed + 1000 + domain_index)
            self.heads[domain] = nn.Linear(64, 1)
            moments = statistics[domain]
            if moments.sd <= 1e-12 or not np.isfinite(moments.sd):
                raise ValueError("Domain output SD is degenerate")
            self.register_buffer(
                f"mean_{domain_index}",
                torch.tensor(moments.mean, dtype=torch.float32),
            )
            self.register_buffer(
                f"sd_{domain_index}",
                torch.tensor(moments.sd, dtype=torch.float32),
            )
        torch.manual_seed(run_seed)

    @property
    def domains(self) -> tuple[str, ...]:
        return self._domain_order

    def forward(
        self, inputs: torch.Tensor, domain_ids: Sequence[str]
    ) -> torch.Tensor:
        if len(domain_ids) != len(inputs):
            raise ValueError("Every CNN row requires one observed domain ID")
        if any(domain not in self.heads for domain in domain_ids):
            raise ValueError("Unknown domain ID")
        encoded = self.encoder(inputs)
        output = torch.empty(
            (len(inputs), 1), dtype=encoded.dtype, device=encoded.device
        )
        for domain in self._domain_order:
            positions = [
                index
                for index, value in enumerate(domain_ids)
                if value == domain
            ]
            if not positions:
                continue
            index = torch.tensor(
                positions, dtype=torch.long, device=encoded.device
            )
            domain_index = DOMAINS.index(domain)
            prediction = self.heads[domain](encoded.index_select(0, index))
            prediction = (
                getattr(self, f"mean_{domain_index}")
                + getattr(self, f"sd_{domain_index}") * prediction
            )
            output.index_copy_(0, index, prediction)
        return output


def parameter_counts(model: DomainAwareCNN) -> dict[str, int]:
    """Return trainable encoder, per-head, and total parameter counts."""
    encoder = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    heads = {
        domain: sum(parameter.numel() for parameter in head.parameters())
        for domain, head in model.heads.items()
    }
    return {
        "encoder": encoder,
        **{f"head_{domain}": count for domain, count in heads.items()},
        "total": encoder + sum(heads.values()),
    }
