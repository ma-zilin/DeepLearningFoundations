"""B3：使用完整 train split 训练状态输入的 MLP Behavior Cloning policy。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BehaviorCloning.dataset import select_episodes
from BehaviorCloning.policy import BCPolicy


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "b3_bc_policy.pt"
HISTORY_PATH = ARTIFACT_DIR / "b3_training_history.json"
LOSS_PLOT_PATH = ARTIFACT_DIR / "b3_train_validation_loss.png"


def _normalized_tensors(
    split_dataset: dict[str, np.ndarray],
    normalization: dict[str, list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    observation_mean = np.asarray(normalization["observation_mean"], dtype=np.float32)
    observation_std = np.asarray(normalization["observation_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)

    observations = (
        split_dataset["observation"] - observation_mean
    ) / observation_std
    actions = (split_dataset["expert_action"] - action_mean) / action_std
    return torch.from_numpy(observations), torch.from_numpy(actions)


def _mean_mse(
    policy: BCPolicy,
    loader: DataLoader,
    loss_function: nn.Module,
) -> float:
    policy.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for observations, target_actions in loader:
            predictions = policy(observations)
            batch_loss = loss_function(predictions, target_actions)
            total_loss += float(batch_loss.item()) * len(observations)
            num_samples += len(observations)
    return total_loss / num_samples


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(3)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(ARTIFACT_DIR / metadata["raw_dataset"]) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}

    split_ids = {
        name: np.asarray(ids, dtype=np.int64)
        for name, ids in metadata["episode_ids"].items()
    }
    train_dataset = select_episodes(dataset, split_ids["train"])
    validation_dataset = select_episodes(dataset, split_ids["validation"])
    train_observations, train_actions = _normalized_tensors(
        train_dataset, metadata["normalization"]
    )
    validation_observations, validation_actions = _normalized_tensors(
        validation_dataset, metadata["normalization"]
    )

    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(train_observations, train_actions),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(4),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_observations, validation_actions),
        batch_size=512,
        shuffle=False,
    )

    model_config = {
        "observation_dim": 3,
        "action_dim": 1,
        "hidden_dims": [64, 64],
    }
    policy = BCPolicy(**model_config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()

    num_epochs = 150
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, num_epochs + 1):
        policy.train()
        total_train_loss = 0.0
        num_train_samples = 0
        for observations, target_actions in train_loader:
            predictions = policy(observations)
            loss = loss_function(predictions, target_actions)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss 出现 NaN 或 Inf")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.detach().item()) * len(observations)
            num_train_samples += len(observations)

        train_loss = total_train_loss / num_train_samples
        validation_loss = _mean_mse(policy, validation_loader, loss_function)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in policy.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("训练过程中没有产生有效 checkpoint")

    policy.load_state_dict(best_state)
    policy.eval()
    probe_raw_observation = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    observation_mean = np.asarray(
        metadata["normalization"]["observation_mean"], dtype=np.float32
    )
    observation_std = np.asarray(
        metadata["normalization"]["observation_std"], dtype=np.float32
    )
    action_mean = np.asarray(
        metadata["normalization"]["action_mean"], dtype=np.float32
    )
    action_std = np.asarray(
        metadata["normalization"]["action_std"], dtype=np.float32
    )
    probe_normalized = (probe_raw_observation - observation_mean) / observation_std
    with torch.no_grad():
        probe_normalized_action = policy(
            torch.from_numpy(probe_normalized).unsqueeze(0)
        ).numpy()[0]
    probe_raw_action = probe_normalized_action * action_std + action_mean

    checkpoint = {
        "model_state_dict": best_state,
        "model_config": model_config,
        "normalization": metadata["normalization"],
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_validation_loss,
        "probe_raw_observation": probe_raw_observation.tolist(),
        "probe_raw_action": probe_raw_action.tolist(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    history = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "checkpoint_metric": "validation_normalized_action_mse",
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_validation_loss,
        "train_normalized_mse": train_losses,
        "validation_normalized_mse": validation_losses,
    }
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    epochs = np.arange(1, num_epochs + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(epochs, train_losses, label="train")
    axis.semilogy(epochs, validation_losses, label="validation")
    axis.axvline(best_epoch, color="black", linestyle="--", label="best checkpoint")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized action MSE")
    axis.set_title("BC policy training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOSS_PLOT_PATH, dpi=160)
    plt.close(figure)

    print(f"train samples: {len(train_observations)}")
    print(f"validation samples: {len(validation_observations)}")
    print(f"best epoch: {best_epoch}")
    print(f"best validation normalized MSE: {best_validation_loss:.8f}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"history: {HISTORY_PATH}")
    print(f"loss plot: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
