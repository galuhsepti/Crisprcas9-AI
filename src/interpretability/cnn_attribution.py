"""
CNN attribution (interpretability) utilities for the CRISPR-Cas9 pipeline.

Provides gradient-based saliency and Integrated Gradients attribution over the
positions of the 30-mer input window. Attribution is computed on the raw
network (``nn.Module``), independent of the ``CNNModel`` prediction wrapper,
so it works on any single-output module.

Interpretation notes
--------------------
- Attribution reflects how a *fitted* model uses the input; it does not by
  itself prove a biological mechanism. Sign is not interpreted as "activates"
  or "represses": the CNN output is a single regression prediction and the raw
  input is a one-hot code, so a position with a high absolute-gradient sum is
  simply a position the model is sensitive to.
- Profiles are normalized per sample (unit L1 over positions) before
  averaging, so every sample contributes equally regardless of its activity
  magnitude.
- Saliency (|gradient|) measures local sensitivity; Integrated Gradients
  sums gradients along a straight path from a zero baseline, giving a
  contribution-style attribution. They are complementary and are cross-checked
  for internal consistency in the Phase 8 analysis.
"""

from typing import Dict, List, Optional
import numpy as np
import torch

from ..ablation.sequence_regions import REGION_POSITIONS


def _to_channel_first(X: np.ndarray) -> torch.Tensor:
    """Convert (n, L, 4) numpy one-hot to a (n, 4, L) float tensor."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3 and X.shape[2] == 4:
        X = X.transpose(0, 2, 1)
    return torch.from_numpy(X.copy())


def position_saliency(
    network: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256
) -> np.ndarray:
    """
    Mean per-position saliency = mean over samples of the per-position sum of
    |d(output)/d(one-hot channel)|, normalized per sample to unit L1.

    Args:
        network: Single-output nn.Module (e.g. future CRISPRsvGN). Must be
            trainable (has ``requires_grad`` parameters) and in eval mode
            where appropriate.
        X: One-hot input (n, context_length, 4)
        batch_size: Forward-pass batch size

    Returns:
        Profile (context_length,) with unit L1 mean.
    """
    network.eval()
    with torch.enable_grad():
        profiles = []
        n = len(X)
        for start in range(0, n, batch_size):
            batch = X[start:start + batch_size]
            x = _to_channel_first(batch)
            x = x.clone().detach().requires_grad_(True)

            out = network(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("Network output contains NaN/Inf during saliency")
            scalar = out.sum()
            grads = torch.autograd.grad(scalar, x, create_graph=False,
                                        retain_graph=False)[0]
            # (batch, 4, L) -> sum abs over channels -> (batch, L)
            g = grads.detach().abs().sum(dim=1).cpu().numpy()
            profiles.append(_normalize_rows(g))

    return np.concatenate(profiles, axis=0).mean(axis=0)


def integrated_gradients(
    network: torch.nn.Module,
    X: np.ndarray,
    steps: int = 50,
    batch_size: int = 256
) -> np.ndarray:
    """
    Mean Integrated Gradients profile over positions.

    For each sample the IG of each unit is
        ``(x - x') * mean_k grad(x' + k/N * (x - x'))``
    with a zero baseline ``x'``. Per-position attribution is the per-sample sum
    of absolute IG over channels, normalized per sample to unit L1.

    Args:
        network: Single-output nn.Module
        X: One-hot input (n, context_length, 4)
        steps: Number of path steps (>= 1)
        batch_size: Forward-pass batch size

    Returns:
        Profile (context_length,) with unit L1 mean.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    profiles = []

    network.eval()
    with torch.enable_grad():
        for start in range(0, n, batch_size):
            batch = X[start:start + batch_size]
            x = _to_channel_first(batch)                 # (b, 4, L)
            ig_acc = torch.zeros_like(x)

            for alpha in np.linspace(0.0, 1.0, steps + 1):
                if alpha == 0.0:
                    continue
                interp = x * alpha                        # zero baseline
                interp = interp.clone().detach().requires_grad_(True)
                out = network(interp)
                scalar = out.sum()
                grads = torch.autograd.grad(
                    scalar, interp, create_graph=False, retain_graph=False)[0]
                ig_acc = ig_acc + grads.detach() / steps

            ig = (x * ig_acc)                             # (b, 4, L) path length x avg grad
            g = ig.abs().sum(dim=1).cpu().numpy()         # (b, L)
            profiles.append(_normalize_rows(g))

    return np.concatenate(profiles, axis=0).mean(axis=0)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L1 normalization with NaN guard."""
    row_sum = np.abs(mat).sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return mat / row_sum


def attribution_by_region(
    profile: np.ndarray,
    context_length: int = 30,
    regions: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Fraction of (unnormalized) attribution mass inside each sequence region.

    Regions are defined by ``REGION_POSITIONS`` (guide [4:24], PAM [24:27],
    5'/3' flanks). Fractions are absolute-importance weighted.

    Args:
        profile: Per-position attribution (must be non-negative, e.g. the
            normalized saliency/IG profiles or plain summed absolute values).
        context_length: Full window length
        regions: Subset of 'guide', 'pam', 'flanks_5p', 'flanks_3p'

    Returns:
        Mapping region name -> fraction of positive attribution mass.
    """
    if len(profile) != context_length:
        raise ValueError(
            f"profile length {len(profile)} != context_length {context_length}"
        )
    if np.any(profile < 0):
        raise ValueError("profile must be non-negative")

    if regions is None:
        regions = ['guide', 'pam', 'flanks_5p', 'flanks_3p']

    region_mask = {
        'guide': REGION_POSITIONS['guide'],
        'pam': list(range(24, 27)),
        'flanks_5p': list(range(0, 4)),
        'flanks_3p': list(range(27, 30)),
    }
    for r in regions:
        if r not in region_mask:
            raise ValueError(f"Unknown region '{r}'")

    total = float(np.sum(profile))
    if total == 0.0:
        return {r: 0.0 for r in regions}

    return {
        r: float(np.sum(profile[region_mask[r]]) / total)
        for r in regions
    }