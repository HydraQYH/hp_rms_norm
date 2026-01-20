import torch


def hp_rms_norm(
    input,
    weight,
    residual,
    eps,
):
    torch.ops.hp_rms_norm.rms_norm.default(
        input,
        weight,
        residual,
        eps,
    )

def relu(
    output,
    input,
):
    torch.ops.hp_rms_norm.relu.default(
        output,
        input,
    )
