"""
Phase 13 embedding + 1D CNN model.

Downstream architecture mirrors the canonical Phase 5 CNN (CRISPRsvGN): three
parallel Conv1d branches (kernels [5,7,9], 64 filters each), ReLU, MaxPool,
AdaptiveAvgPool, concatenation, Dense(64) + ReLU + Dropout(0.3), output(1).

The ONLY change from the canonical model is the input representation: instead
of a one-hot (n, 30, 4) tensor followed by Conv1d with 4 input channels, the
input is an integer token index tensor (n, 30) for the nucleotide-embedding
representation (or (n, seq_len) for the k-mer representation), which is mapped
through a learned nn.Embedding to (n, seq_len, embedding_dim), then transposed
to channel-first (n, embedding_dim, seq_len) for the SAME Conv1d branches.

To keep capacity comparable to the canonical CNN (4 input channels, 64 filters),
the embedding dimension controls the Conv1d input channels. Default embedding
dim = 8 keeps downstream capacity similar (8 channels vs 4 channels x 2).

Training protocol matches the canonical CNN exactly: Adam, lr 0.001, batch 32,
MSE loss, early stopping on validation loss with patience 10, max 100 epochs,
fixed seed 42. Validation is used only for early stopping / model selection;
the external test set is NEVER used for training, tuning, or selection.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


class EmbeddingCNN(nn.Module):
    """
    Learned-embedding CNN for sgRNA activity regression.

    Input: integer token index tensor of shape (batch, seq_len).
    - nn.Embedding v (vocab_size, embedding_dim) -> (batch, seq_len, embedding_dim)
    - permute to (batch, embedding_dim, seq_len)
    - three parallel Conv1d branches: same structure as canonical CNN.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        embedding_dim: int = 8,
        seq_len: int = 30,
        conv_n_filters: Any = 64,
        conv_kernel_sizes: List[int] = [5, 7, 9],
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        n_classes: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        if isinstance(conv_n_filters, int):
            conv_n_filters = [conv_n_filters] * len(conv_kernel_sizes)
        self.conv_branches = nn.ModuleList()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        n_conv_outputs = 0
        for k, n_filters in zip(conv_kernel_sizes, conv_n_filters):
            n_conv_outputs += n_filters
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, n_filters, kernel_size=k,
                              padding=k // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
            )

        self.dense1 = nn.Linear(n_conv_outputs, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: integer indices (batch, seq_len) -> (batch, 1)
        """
        emb = self.embedding(x)             # (batch, seq_len, embedding_dim)
        emb = emb.permute(0, 2, 1)          # (batch, embedding_dim, seq_len)

        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(emb)
            out = self.global_pool(out)
            branch_outputs.append(out.squeeze(-1))
        conv_out = torch.cat(branch_outputs, dim=1)

        dense = self.dropout(self.relu(self.dense1(conv_out)))
        return self.output(dense)


class EmbeddingCNNModel:
    """
    Wrapper providing the same sklearn-like interface as CNNModel, but taking
    integer token index arrays as input instead of one-hot arrays.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        embedding_dim: int = 8,
        seq_len: int = 30,
        conv_n_filters: Any = 64,
        conv_kernel_sizes: List[int] = [5, 7, 9],
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        optimizer: str = "adam",
        loss: str = "mse",
        random_state: int = 42,
        use_cpu_threads: int = 4,
    ):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if use_cpu_threads > 0:
            torch.set_num_threads(use_cpu_threads)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.conv_n_filters = conv_n_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.optimizer_name = optimizer
        self.loss_name = loss
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = EmbeddingCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            conv_n_filters=conv_n_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            n_classes=1,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_history: Dict[str, Any] = {}
        self.is_fitted = False

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected token input (n, {self.seq_len}), got {X.shape}"
            )
        Xt = torch.from_numpy(X).to(self.device)
        yt = None
        if y is not None:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            yt = torch.from_numpy(y).to(self.device)
        return Xt, yt

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        Xt, yt = self._to_tensor(X_train, y_train)
        train_dataset = TensorDataset(Xt, yt)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xvt, yvt = self._to_tensor(X_val, y_val)
            val_dataset = TensorDataset(Xvt, yvt)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()
        best_val_loss = float("inf")
        best_epoch = None
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.network(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)

            avg_val_loss = None
            if val_loader is not None:
                self.network.eval()
                val_sum = 0.0
                v_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = self.network(xb)
                        loss = self.criterion(pred, yb)
                        val_sum += loss.item()
                        v_batches += 1
                avg_val_loss = val_sum / v_batches
                history["val_loss"].append(avg_val_loss)

            if verbose:
                val_str = f" | val_loss: {avg_val_loss:.4f}" if avg_val_loss is not None else ""
                print(f"Epoch {epoch}/{self.epochs} | train_loss: {avg_train_loss:.4f}{val_str}")

            if val_loader is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (patience {self.patience})")
                        break

        if best_state is not None:
            self.network.load_state_dict(best_state)

        total_epochs_run = len(history["train_loss"])
        if val_loader is not None:
            assert best_epoch is not None and best_epoch <= total_epochs_run
            assert np.isclose(history["val_loss"][best_epoch - 1], best_val_loss)
        else:
            best_epoch = total_epochs_run
            best_val_loss = None

        self.best_val_loss = best_val_loss
        self.training_history = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs_run": total_epochs_run,
            "early_stopping": val_loader is not None,
            "training_time": time.time() - start_time,
        }
        self.is_fitted = True
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        self.network.eval()
        Xt, _ = self._to_tensor(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                batch = Xt[i:i + self.batch_size]
                preds.append(self.network(batch).cpu().numpy())
        return np.concatenate(preds).ravel()

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "hyperparameters": self.get_params(),
                "training_history": self.training_history,
                "embedding_dim": self.embedding_dim,
                "vocab_size": self.vocab_size,
                "seq_len": self.seq_len,
            },
            filepath,
        )

    @classmethod
    def load_model(cls, filepath: str) -> "EmbeddingCNNModel":
        ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparameters"]
        instance = cls(**hp)
        instance.network.load_state_dict(ckpt["state_dict"])
        instance.training_history = ckpt["training_history"]
        instance.is_fitted = True
        return instance

    def get_params(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "seq_len": self.seq_len,
            "conv_n_filters": self.conv_n_filters,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "optimizer": self.optimizer_name,
            "loss": self.loss_name,
            "random_state": self.random_state,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.network.parameters())
        embedding_params = self.network.embedding.weight.numel()
        return {
            "total_trainable_parameters": int(total),
            "embedding_parameters": int(embedding_params),
            "downstream_parameters": int(total - embedding_params),
        }
