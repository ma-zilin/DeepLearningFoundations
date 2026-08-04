"""按完整 episode 构造 B2 split，并保存 train-only 标准化统计量。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from BehaviorCloning.dataset import (
    compute_normalization_statistics,
    select_episodes,
    split_episode_ids,
    validate_expert_dataset,
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RAW_DATASET_PATH = ARTIFACT_DIR / "b1_expert_episodes.npz"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"


def _split_summary(split_dataset: dict[str, np.ndarray]) -> dict[str, object]:
    """汇总一个 split 的 episode、transition 和数值范围。"""
    observations = split_dataset["observation"]
    actions = split_dataset["expert_action"]
    return {
        "num_episodes": int(len(np.unique(split_dataset["episode_id"]))),
        "num_transitions": int(len(split_dataset["episode_id"])),
        "observation_min": observations.min(axis=0).tolist(),
        "observation_max": observations.max(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
    }


def main() -> None:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"找不到 B1 原始数据: {RAW_DATASET_PATH}，请先运行 collect_expert_data"
        )

    with np.load(RAW_DATASET_PATH) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}
    validate_expert_dataset(dataset, expected_num_episodes=100)

    split_seed = 1
    split_ids = split_episode_ids(dataset, seed=split_seed)
    split_datasets = {
        name: select_episodes(dataset, episode_ids)
        for name, episode_ids in split_ids.items()
    }
    statistics = compute_normalization_statistics(split_datasets["train"])

    metadata = {
        "raw_dataset": RAW_DATASET_PATH.name,
        "dataset_seed": 0,
        "split_seed": split_seed,
        "episode_ids": {
            name: episode_ids.tolist() for name, episode_ids in split_ids.items()
        },
        "normalization": {
            key: value.tolist() for key, value in statistics.items()
        },
        "summary": {
            name: _split_summary(split_dataset)
            for name, split_dataset in split_datasets.items()
        },
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"metadata: {METADATA_PATH}")
    for name in ("train", "validation", "test"):
        summary = metadata["summary"][name]
        print(
            f"{name}: episodes={summary['num_episodes']}, "
            f"transitions={summary['num_transitions']}, "
            f"observation_range="
            f"{summary['observation_min']} -> {summary['observation_max']}, "
            f"action_range={summary['action_min']} -> {summary['action_max']}"
        )
    print(f"normalization (train only): {metadata['normalization']}")


if __name__ == "__main__":
    main()
