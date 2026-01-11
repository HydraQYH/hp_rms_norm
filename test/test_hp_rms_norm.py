import random
import torch
import hp_rms_norm


def ref_rms_norm(
    inp: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    eps: float
):
  return res + torch.rms_norm(
    inp,
    normalized_shape=(inp.shape[-1],),
    weight=weight,
    eps=eps
  )

def check_diff(tokens: int, hidden_dim: int, eps: float=0.0, dtype: torch.dtype=torch.bfloat16):
  if dtype == torch.bfloat16:
    rtol = 1.6e-2
    atol = 1e-3
  elif dtype == torch.float16:
    rtol = 1e-3
    atol = 1e-5

  inp = torch.normal(0, 0.1, (tokens, hidden_dim), dtype=dtype, device='cuda')
  res = torch.normal(0, 0.1, (tokens, hidden_dim), dtype=dtype, device='cuda')
  weight = torch.normal(0, 0.1, (hidden_dim,), dtype=dtype, device='cuda')
  out = torch.zeros_like(inp)
  ref_out = ref_rms_norm(inp, res, weight, eps)
  hp_rms_norm.hp_rms_norm(out, inp, weight, res, eps)
  torch.cuda.synchronize()
  torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)

if __name__ == '__main__':
  for dtype in [torch.bfloat16, torch.float16]:
    for hidden_dim in range(128, 32768 + 128, 128):
      tokens = random.randint(1, 4096)
      check_diff(tokens, hidden_dim, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
      print(f"Check tokens {tokens}, hidden_dim {hidden_dim}, dtype {dtype} Done!")
  # check_diff(4096, 8192, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
  # check_diff(4096, 16384, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
