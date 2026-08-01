"""二维双峰高斯数据的公共生成函数。"""

import torch


CENTERS = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])


def sample_bimodal_components(
    num_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样峰编号和二维标准高斯噪声。"""
    mode_ids = torch.randint(
        low=0,
        high=2,
        size=(num_samples,),
        generator=generator,
    )
    standard_noise = torch.randn(
        (num_samples, 2),
        generator=generator,
    )
    return mode_ids, standard_noise


def build_bimodal_samples(
    mode_ids: torch.Tensor,
    standard_noise: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """将标准高斯噪声平移到对应峰中心。"""
    if standard_noise.shape != (mode_ids.shape[0], 2):
        raise ValueError(
            "standard_noise 必须具有形状 [num_samples, 2]，"
            f"实际为 {list(standard_noise.shape)}"
        )
    return CENTERS[mode_ids] + std * standard_noise


def generate_bimodal_data(
    num_samples: int,
    std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次生成双峰样本及其峰编号。"""
    mode_ids, standard_noise = sample_bimodal_components(
        num_samples=num_samples,
        generator=generator,
    )
    samples = build_bimodal_samples(
        mode_ids=mode_ids,
        standard_noise=standard_noise,
        std=std,
    )
    return samples, mode_ids
