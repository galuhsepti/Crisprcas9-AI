"""
Phase 13 configuration.

This configuration is PRE-REGISTERED: it is frozen before training and is NOT
selected using Moreno-Mateos. Every value here is chosen to keep the Phase 13
model as close as possible to the canonical Phase 5 CNN while swapping ONLY the
input representation (one-hot -> learned nucleotide embedding).

Pre-registration policy:
- Use the canonical DeepSpCas9 dataset (data/raw/DeepSpCas9.csv).
- Use the exact canonical split (train_test_split test_size=0.15, seed 42).
- Keep downstream architecture, optimizer, learning rate, batch size, epoch
  budget, patience, loss, and early-stopping rule identical to the canonical
  CNN wherever possible.
- Choose ONE primary representation BEFORE training, and a pre-registered
  selection rule if a second representation is evaluated.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass(frozen=True)
class Phase13ExperimentConfig:
    # ---- Dataset / split (MUST match canonical Phase 5) ----
    project_root: str = str(Path(__file__).resolve().parents[2])
    primary_dataset: str = "data/raw/DeepSpCas9.csv"
    external_dataset: str = "data/raw/Moreno-Mateos.csv"
    context_length: int = 30
    val_ratio: float = 0.15
    split_seed: int = 42

    # ---- Representation (PRE-REGISTERED primary) ----
    # Primary candidate: learned nucleotide (character) embedding of each of the
    # 30 positions. Vocabulary is the four canonical DNA nucleotides {A,C,G,T}.
    # A learned embedding vector is fitted per nucleotide on TRAIN ONLY.
    representation_name: str = "nucleotide_embedding"
    vocabulary: List[str] = field(default_factory=lambda: ["A", "C", "G", "T"])
    embedding_dim: int = 8
    nuc_to_index: Dict[str, int] = field(
        default_factory=lambda: {"A": 0, "C": 1, "G": 2, "T": 3}
    )

    # Secondary pre-registered candidate (optional, for a controlled second
    # representation). Overlapping 3-mer tokens -> learned embedding. Selection
    # between the two uses the CANONICAL validation protocol only.
    kmer_representation: bool = False
    kmer_size: int = 3
    kmer_vocabulary: List[str] = field(default_factory=list)

    # ---- Downstream architecture (match canonical Phase 5 where compatible) --
    conv_n_filters: int = 64
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [5, 7, 9])
    dense_units: int = 64
    dropout_rate: float = 0.3
    output_units: int = 1

    # ---- Training protocol (identical to canonical Phase 5) ----
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    optimizer: str = "adam"
    loss: str = "mse"
    random_seed: int = 42
    use_cpu_threads: int = 4

    # ---- Bootstrap reproducibility ----
    n_boot: int = 1000
    bootstrap_seed: int = 20260906

    # ---- Activity bins (FIXED, same grid as earlier phases) ----
    activity_edges: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # ---- Artifact paths ----
    canonical_checkpoint: str = "models/cnn_baseline_20260905_011720.pt"
    output_dir: str = "results/experiments"
    figures_dir: str = "results/figures"
    models_dir: str = "models"
    predictions_dir: str = "results/predictions"

    def paths(self) -> Dict[str, Path]:
        root = Path(self.project_root)
        return {
            "primary_dataset": root / self.primary_dataset,
            "external_dataset": root / self.external_dataset,
            "canonical_checkpoint": root / self.canonical_checkpoint,
            "output_dir": root / self.output_dir,
            "figures_dir": root / self.figures_dir,
            "models_dir": root / self.models_dir,
            "predictions_dir": root / self.predictions_dir,
        }


# Default pre-registered configuration.
REPRESENTATION_AUDIT_CONFIG = Phase13ExperimentConfig()
