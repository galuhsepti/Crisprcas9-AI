"""Deterministic batching and CNN training mechanics for locked Phase 18C."""

from __future__ import annotations

import copy
import ctypes
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from src.experiment_protocols.multidomain_protocol import DOMAINS
from src.multidomain.cnn import DomainAwareCNN, set_deterministic_runtime
from src.multidomain.data import TrainingStatistics


class TrainingInstability(RuntimeError):
    """Numerical failure carrying the completed partial training history."""

    def __init__(self, message, history, exposures):
        super().__init__(message)
        self.history = copy.deepcopy(history)
        self.exposures = dict(exposures)


def process_peak_memory_bytes() -> int | None:
    """Return peak process working-set memory using platform stdlib APIs."""
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.peak_working_set_size) if success else None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):
        return None


def deterministic_epoch_batches(
    sorted_ids: Sequence[str],
    batch_size: int,
    steps: int,
    run_seed: int,
    domain_index: int,
    epoch: int,
) -> np.ndarray:
    """Generate locked full batches with deterministic exhaustion cycles."""
    identifiers = np.asarray(sorted_ids)
    if (
        identifiers.ndim != 1
        or identifiers.size == 0
        or len(set(identifiers.tolist())) != identifiers.size
    ):
        raise ValueError("Sampling IDs must be nonempty and unique")
    if identifiers.tolist() != sorted(identifiers.tolist()):
        raise ValueError("Sampling IDs must be sorted")
    if batch_size <= 0 or steps <= 0 or epoch < 1:
        raise ValueError(
            "Batch size, steps, and one-indexed epoch must be positive"
        )
    if domain_index not in range(len(DOMAINS)):
        raise ValueError("Unknown domain index")
    required = batch_size * steps
    sampled = []
    cycle = 0
    while len(sampled) < required:
        generator = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([run_seed, domain_index, epoch, cycle])
            )
        )
        permutation = generator.permutation(identifiers.size)
        take = min(required - len(sampled), identifiers.size)
        sampled.extend(permutation[:take].tolist())
        cycle += 1
    return np.asarray(sampled, dtype=np.int64).reshape(steps, batch_size)


def normalized_mse(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    statistics: TrainingStatistics,
) -> torch.Tensor:
    """Compute raw-target MSE divided by train-only population variance."""
    if statistics.variance <= 1e-24 or not np.isfinite(statistics.variance):
        raise ValueError("Training variance is degenerate")
    return torch.mean((predictions.reshape(-1) - labels.reshape(-1)) ** 2) / (
        statistics.variance
    )


@dataclass
class CheckpointController:
    """Separate exact-minimum retention from significant patience resets."""

    patience: int = 10
    min_improvement: float = 0.0001

    def __post_init__(self):
        if self.patience <= 0 or self.min_improvement <= 0:
            raise ValueError(
                "Patience and minimum improvement must be positive"
            )
        self.best_loss = math.inf
        self.significant_best = math.inf
        self.best_epoch = None
        self.best_state = None
        self.non_reset_epochs = 0

    def observe(self, epoch: int, loss: float, state: Mapping) -> bool:
        if epoch < 1 or not np.isfinite(loss):
            raise ValueError(
                "Checkpoint objective must be finite and one-indexed"
            )
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(state)
        significant = (
            not np.isfinite(self.significant_best)
            or loss < self.significant_best - self.min_improvement
        )
        if significant:
            self.significant_best = float(loss)
            self.non_reset_epochs = 0
        else:
            self.non_reset_epochs += 1
        return self.non_reset_epochs >= self.patience


def _validation_objective(
    model: DomainAwareCNN,
    features: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    statistics: Mapping[str, TrainingStatistics],
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = {}
    with torch.no_grad():
        for domain in model.domains:
            inputs = torch.as_tensor(features[domain], dtype=torch.float32)
            targets = torch.as_tensor(labels[domain], dtype=torch.float32)
            predictions = model(inputs, [domain] * len(inputs))
            value = normalized_mse(predictions, targets, statistics[domain])
            losses[domain] = float(value.item())
    objective = float(np.mean([losses[domain] for domain in model.domains]))
    if not np.isfinite(objective):
        raise RuntimeError("Nonfinite validation objective")
    return objective, losses


def train_cnn(
    model: DomainAwareCNN,
    train_features: Mapping[str, np.ndarray],
    train_labels: Mapping[str, np.ndarray],
    train_ids: Mapping[str, Sequence[str]],
    validation_features: Mapping[str, np.ndarray],
    validation_labels: Mapping[str, np.ndarray],
    statistics: Mapping[str, TrainingStatistics],
    run_seed: int,
    *,
    steps_per_epoch: int = 467,
    max_epochs: int = 100,
    progress_callback=None,
) -> dict:
    """Fit one locked CNN arm while callers control the evaluation freeze."""
    if set(model.domains) != set(train_features):
        raise ValueError("Training domains do not match model heads")
    set_deterministic_runtime(run_seed)
    model.cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    controller = CheckpointController()
    history = []
    exposures = {domain: 0 for domain in model.domains}
    per_domain_batch = 16 if len(model.domains) == 2 else 32
    start = time.perf_counter()
    tracemalloc.start()

    def fail(message, epoch, step=None):
        event = {
            "status": "UNSTABLE_RESULT",
            "reason": message,
            "epoch": epoch,
            "step": step,
            "completed_epochs": len(history),
            "domain_exposures": dict(exposures),
        }
        if progress_callback is not None:
            progress_callback(event)
        tracemalloc.stop()
        raise TrainingInstability(message, history, exposures)

    for epoch in range(1, max_epochs + 1):
        batches = {
            domain: deterministic_epoch_batches(
                train_ids[domain],
                per_domain_batch,
                steps_per_epoch,
                run_seed,
                DOMAINS.index(domain),
                epoch,
            )
            for domain in model.domains
        }
        model.train()
        epoch_losses = {domain: [] for domain in model.domains}
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            losses = []
            for domain in model.domains:
                indices = batches[domain][step]
                inputs = torch.as_tensor(
                    train_features[domain][indices], dtype=torch.float32
                )
                targets = torch.as_tensor(
                    train_labels[domain][indices], dtype=torch.float32
                )
                predictions = model(inputs, [domain] * len(inputs))
                loss = normalized_mse(predictions, targets, statistics[domain])
                if not torch.isfinite(loss):
                    fail("Nonfinite training objective", epoch, step + 1)
                losses.append(loss)
                epoch_losses[domain].append(float(loss.item()))
                exposures[domain] += len(indices)
            objective = torch.stack(losses).mean()
            objective.backward()
            optimizer.step()
        try:
            validation, validation_domains = _validation_objective(
                model,
                validation_features,
                validation_labels,
                statistics,
            )
        except RuntimeError as error:
            fail(str(error), epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": {
                    domain: float(np.mean(epoch_losses[domain]))
                    for domain in model.domains
                },
                "validation_loss": validation_domains,
                "validation_objective": validation,
            }
        )
        if progress_callback is not None:
            progress_callback({"status": "EPOCH_COMPLETE", **history[-1]})
        if controller.observe(epoch, validation, model.state_dict()):
            break
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if controller.best_state is None:
        raise RuntimeError("No finite CNN checkpoint was retained")
    model.load_state_dict(controller.best_state)
    return {
        "history": history,
        "best_epoch": controller.best_epoch,
        "best_validation_objective": controller.best_loss,
        "epochs_run": len(history),
        "optimizer_steps": len(history) * steps_per_epoch,
        "domain_exposures": exposures,
        "training_seconds": time.perf_counter() - start,
        "python_tracemalloc_peak_bytes": peak_memory,
        "process_peak_working_set_bytes": process_peak_memory_bytes(),
    }
