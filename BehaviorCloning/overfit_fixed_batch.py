"""B3-A：用固定小 batch 验证 BC 数据流、loss 和梯度。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from BehaviorCloning.dataset import select_episodes
from BehaviorCloning.policy import BCPolicy


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"
LOSS_PLOT_PATH = ARTIFACT_DIR / "b3_fixed_batch_overfit_loss.png"


def main() -> None:
    torch.set_num_threads(1)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raw_dataset_path = ARTIFACT_DIR / metadata["raw_dataset"]
    with np.load(raw_dataset_path) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}

    train_episode_ids = np.asarray(metadata["episode_ids"]["train"], dtype=np.int64)
    train_dataset = select_episodes(dataset, train_episode_ids)

    normalization = metadata["normalization"]
    observation_mean = np.asarray(normalization["observation_mean"], dtype=np.float32)
    observation_std = np.asarray(normalization["observation_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)

    batch_size = 128
    batch_seed = 2
    batch_indices = np.random.default_rng(batch_seed).choice(
        len(train_dataset["observation"]),
        size=batch_size,
        replace=False,
    )
    observation_batch = (
        train_dataset["observation"][batch_indices] - observation_mean
    ) / observation_std
    action_batch = (
        train_dataset["expert_action"][batch_indices] - action_mean
    ) / action_std

    observations = torch.from_numpy(observation_batch)
    target_actions = torch.from_numpy(action_batch)
    if observations.shape != (batch_size, 3):
        raise RuntimeError(f"unexpected observation shape: {observations.shape}")
    if target_actions.shape != (batch_size, 1):
        raise RuntimeError(f"unexpected action shape: {target_actions.shape}")

    torch.manual_seed(2)
    policy = BCPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()

    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    losses: list[float] = []
    nonzero_gradient_elements = 0
    gradient_norm = 0.0

    num_steps = 3000
    policy.train()
    for _ in range(num_steps):
        predicted_actions = policy(observations)
        loss = loss_function(predicted_actions, target_actions)
        if not torch.isfinite(loss):
            raise RuntimeError("fixed-batch loss 出现 NaN 或 Inf")

        optimizer.zero_grad()
        loss.backward()

        nonzero_gradient_elements = sum(
            int(torch.count_nonzero(parameter.grad).item())
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
        gradient_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(parameter.grad.detach() ** 2)
                    for parameter in policy.parameters()
                    if parameter.grad is not None
                )
            ).item()
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))

    parameters_changed = all(
        not torch.equal(initial_parameters[name], parameter.detach())
        for name, parameter in policy.named_parameters()
    )
    if not parameters_changed:
        raise RuntimeError("至少有一个参数张量没有发生变化")
    if nonzero_gradient_elements == 0 or not np.isfinite(gradient_norm):
        raise RuntimeError("没有得到有效的有限梯度")

    policy.eval()
    with torch.no_grad():
        final_predictions = policy(observations)
        final_loss = float(loss_function(final_predictions, target_actions).item())

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(np.arange(1, num_steps + 1), losses)
    axis.set_xlabel("optimization step")
    axis.set_ylabel("fixed-batch normalized action MSE")
    axis.set_title("BC policy fixed-batch overfit")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(LOSS_PLOT_PATH, dpi=160)
    plt.close(figure)

    print(f"observation shape: {tuple(observations.shape)}")
    print(f"target action shape: {tuple(target_actions.shape)}")
    print(f"prediction shape: {tuple(final_predictions.shape)}")
    print(f"initial loss: {losses[0]:.8f}")
    print(f"final loss: {final_loss:.8f}")
    print(f"loss reduction: {losses[0] / final_loss:.2f}x")
    print(f"nonzero gradient elements: {nonzero_gradient_elements}")
    print(f"gradient norm: {gradient_norm:.8f}")
    print(f"all parameter tensors changed: {parameters_changed}")
    print(f"loss plot: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
