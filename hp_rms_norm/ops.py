import torch


def hp_rms_norm(
    output,
    input,
    weight,
    residual,
    eps,
):
    torch.ops.hp_rms_norm.rms_norm.default(
        output,
        input,
        weight,
        residual,
        eps,
    )
