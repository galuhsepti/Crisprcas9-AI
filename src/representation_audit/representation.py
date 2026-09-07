"""
Phase 13 sequence representations.

Deterministic tokenization of 30-mer DNA sequences into integer index arrays for
a learned embedding, plus vocabulary definition and token-count statistics.

Guarantees:
- The vocabulary is derived from the canonical four DNA nucleotides {A,C,G,T}.
- Tokenization is a deterministic, fixed transformation: it does NOT depend on
  any external test set, does NOT depend on the training data distribution, and
  does NOT leak any external information.
- All LEARNED parameters (embedding vectors) are fitted on TRAIN ONLY.
"""

from typing import List, Optional, Sequence, Dict

import numpy as np

# Fixed canonical DNA alphabet (order determines the index mapping).
DEFAULT_VOCABULARY = ["A", "C", "G", "T"]
DEFAULT_NUC_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


class NucleotideEmbeddingRepresentation:
    """
    Learned per-nucleotide (character) embedding of the 30-mer.

    Each of the 30 positions is mapped to the index of its nucleotide
    (A/C/G/T) via a deterministic vocabulary. A learned embedding tensor of
    shape (4, embedding_dim) is then indexed by position -> a 30 x embedding_dim
    input tensor for the 1D CNN.

    Tokenization is a fixed deterministic mapping shared by train/val/external.
    The embedding vectors themselves are fitted on TRAIN ONLY by the model.
    """

    name = "nucleotide_embedding"

    def __init__(
        self,
        context_length: int = 30,
        vocabulary: Optional[Sequence[str]] = None,
        nuc_to_index: Optional[Dict[str, int]] = None,
        embedding_dim: int = 8,
    ):
        if vocabulary is None:
            vocabulary = DEFAULT_VOCABULARY
        if nuc_to_index is None:
            nuc_to_index = DEFAULT_NUC_TO_INDEX
        self.context_length = context_length
        self.vocabulary = list(vocabulary)
        self.nuc_to_index = dict(nuc_to_index)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(self.vocabulary)

        # Sanity: vocabulary chars must each have a valid index, and indices
        # must be contiguous covering [0, vocab_size).
        for nuc in self.vocabulary:
            if nuc not in self.nuc_to_index:
                raise ValueError(f"Nucleotide '{nuc}' missing from nuc_to_index")
        expected = set(range(self.vocab_size))
        if set(self.nuc_to_index.values()) != expected:
            raise ValueError(
                "nuc_to_index values must be a permutation of "
                f"0..{self.vocab_size - 1}, got {set(self.nuc_to_index.values())}"
            )

    def tokenize(self, sequence: str) -> np.ndarray:
        """
        Deterministically tokenize a single 30-mer into integer indices.

        Returns an int64 array of shape (context_length,). Any character not in
        the vocabulary raises a ValueError so that ambiguous handling is
        impossible (no silent default).
        """
        sequence = sequence.upper()
        if len(sequence) != self.context_length:
            raise ValueError(
                f"Expected sequence length {self.context_length}, got {len(sequence)}"
            )
        tokens = np.empty(self.context_length, dtype=np.int64)
        for i, ch in enumerate(sequence):
            if ch not in self.nuc_to_index:
                raise ValueError(
                    f"Invalid nucleotide '{ch}' at position {i} in {sequence}. "
                    f"Vocabulary is {self.vocabulary}."
                )
            tokens[i] = self.nuc_to_index[ch]
        return tokens

    def tokenize_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Tokenize a batch of sequences.

        Returns an int64 array of shape (n, context_length). Does not fit any
        parameter; it is a pure deterministic mapping.
        """
        return np.stack([self.tokenize(s) for s in sequences], axis=0)

    def token_statistics(self, sequences: Sequence[str]) -> Dict[str, float]:
        """
        Character frequency statistics over a set of sequences (train-only use).

        This is DESCRIPTIVE metadata for the representation audit; it does not
        influence tokenization, which is fixed.
        """
        tokens = self.tokenize_batch(sequences)
        counts = np.bincount(tokens.ravel(), minlength=self.vocab_size)
        total = counts.sum()
        return {
            f"token_{nuc}_count": int(counts[i])
            for i, nuc in enumerate(self.vocabulary)
        } | {"total_tokens": int(total)}

    def input_dim_per_position(self) -> int:
        """Number of input features per sequence position (embedding dim)."""
        return self.embedding_dim


class KMerEmbeddingRepresentation:
    """
    Pre-registered secondary candidate: overlapping k-mer token embedding.

    Overlapping k-mers of the 30-mer are mapped to a deterministic lexicographic
    index over the fixed {A,C,G,T} alphabet, then embedded. This is NOT the
    primary candidate and is only enabled when kmer_representation=True with a
    pre-registered selection rule (see config).
    """

    name = "kmer_embedding"

    def __init__(
        self,
        k: int = 3,
        context_length: int = 30,
        vocabulary: Optional[Sequence[str]] = None,
        embedding_dim: int = 8,
    ):
        if vocabulary is None:
            vocabulary = DEFAULT_VOCABULARY
        self.k = k
        self.context_length = context_length
        self.vocabulary = list(vocabulary)
        self.nuc_to_index = {nuc: i for i, nuc in enumerate(self.vocabulary)}
        self.embedding_dim = embedding_dim
        self.vocab_size = len(self.vocabulary) ** self.k
        # Number of overlapping k-mers in a sequence of length context_length.
        self.seq_len = context_length - k + 1
        self._kmer_to_index = self._build_kmer_index()

    def _build_kmer_index(self) -> Dict[str, int]:
        """Build deterministic lexicographic kmer->index mapping."""
        mapping = {}
        for idx in range(self.vocab_size):
            code = idx
            chars = []
            for _ in range(self.k):
                chars.append(self.vocabulary[code % len(self.vocabulary)])
                code //= len(self.vocabulary)
            # least-significant nucleotide first; reverse for natural order
            kmer = "".join(reversed(chars))
            mapping[kmer] = idx
        return mapping

    def tokenize(self, sequence: str) -> np.ndarray:
        sequence = sequence.upper()
        if len(sequence) != self.context_length:
            raise ValueError(
                f"Expected sequence length {self.context_length}, got {len(sequence)}"
            )
        tokens = np.empty(self.seq_len, dtype=np.int64)
        for i in range(self.seq_len):
            kmer = sequence[i:i + self.k]
            if kmer not in self._kmer_to_index:
                raise ValueError(f"Invalid k-mer '{kmer}' at position {i}")
            tokens[i] = self._kmer_to_index[kmer]
        return tokens

    def tokenize_batch(self, sequences: Sequence[str]) -> np.ndarray:
        return np.stack([self.tokenize(s) for s in sequences], axis=0)

    def input_dim_per_position(self) -> int:
        return self.embedding_dim
