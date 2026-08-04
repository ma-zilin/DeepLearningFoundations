"""B3：在 validation/test expert observations 上比较 action prediction 基线。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from BehaviorCloning.dataset import select_episodes
from BehaviorCloning.policy import BCPolicy


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "b3_bc_policy.pt"
RESULT_PATH = ARTIFACT_DIR / "b3_open_loop_results.json"


def _predict_raw_actions(
    policy: BCPolicy,
    raw_observations: np.ndarray,
    normalization: dict[str, list[float]],
) -> np.ndarray:
    observation_mean = np.asarray(normalization["observation_mean"], dtype=np.float32)
    observation_std = np.asarray(normalization["observation_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    normalized_observations = (raw_observations - observation_mean) / observation_std

    policy.eval()
    with torch.no_grad():
        normalized_actions = policy(torch.from_numpy(normalized_observations)).numpy()
    return normalized_actions * action_std + action_mean


def _mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def main() -> None:
    torch.set_num_threads(1)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(ARTIFACT_DIR / metadata["raw_dataset"]) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    trained_policy = BCPolicy(**checkpoint["model_config"])
    trained_policy.load_state_dict(checkpoint["model_state_dict"])

    reloaded_probe_action = _predict_raw_actions(
        trained_policy,
        np.asarray(checkpoint["probe_raw_observation"], dtype=np.float32)[None, :],
        checkpoint["normalization"],
    )[0]
    np.testing.assert_allclose(
        reloaded_probe_action,
        np.asarray(checkpoint["probe_raw_action"]),
        rtol=0.0,
        atol=1e-7,
    )

    torch.manual_seed(3)
    untrained_policy = BCPolicy(**checkpoint["model_config"])
    action_mean = np.asarray(
        checkpoint["normalization"]["action_mean"], dtype=np.float32
    )

    results: dict[str, dict[str, float | int]] = {}
    for split_name in ("validation", "test"):
        episode_ids = np.asarray(
            metadata["episode_ids"][split_name], dtype=np.int64
        )
        split_dataset = select_episodes(dataset, episode_ids)
        observations = split_dataset["observation"]
        target_actions = split_dataset["expert_action"]

        zero_predictions = np.zeros_like(target_actions)
        mean_predictions = np.broadcast_to(action_mean, target_actions.shape)
        untrained_predictions = _predict_raw_actions(
            untrained_policy, observations, checkpoint["normalization"]
        )
        trained_predictions = _predict_raw_actions(
            trained_policy, observations, checkpoint["normalization"]
        )
        results[split_name] = {
            "num_transitions": int(len(observations)),
            "zero_action_mse": _mse(zero_predictions, target_actions),
            "train_mean_action_mse": _mse(mean_predictions, target_actions),
            "untrained_mlp_mse": _mse(untrained_predictions, target_actions),
            "trained_bc_mse": _mse(trained_predictions, target_actions),
        }

    RESULT_PATH.write_text(
        json.dumps(
            {
                "checkpoint_best_epoch": checkpoint["best_epoch"],
                "probe_raw_observation": np.asarray(
                    checkpoint["probe_raw_observation"]
                ).tolist(),
                "reloaded_probe_raw_action": reloaded_probe_action.tolist(),
                "physical_action_mse": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "checkpoint reload probe: "
        f"observation={np.asarray(checkpoint['probe_raw_observation']).tolist()}, "
        f"action={reloaded_probe_action.tolist()}"
    )
    for split_name, split_results in results.items():
        print(f"{split_name}: {split_results}")
    print(f"results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
