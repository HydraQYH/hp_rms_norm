import random
import torch
import hp_rms_norm
from flashinfer.norm import fused_add_rmsnorm

def ref_rms_norm(
    inp: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    eps: float
):
  inp_res = inp + res
  out = torch.rms_norm(
    inp_res,
    normalized_shape=(inp.shape[-1],),
    weight=weight,
    eps=eps
  )
  return out, inp_res

def check_diff(tokens: int, hidden_dim: int, eps: float=0.0, dtype: torch.dtype=torch.bfloat16):
  if dtype == torch.bfloat16:
    rtol = 1.6e-2
    atol = 1e-3
  elif dtype == torch.float16:
    rtol = 1e-3
    atol = 1e-5

  inp_ref = torch.normal(0, 0.1, (tokens, hidden_dim), dtype=dtype, device='cuda')
  res_ref = torch.normal(0, 0.1, (tokens, hidden_dim), dtype=dtype, device='cuda')
  weight = torch.normal(0, 0.1, (hidden_dim,), dtype=dtype, device='cuda')

  inp = torch.empty_like(inp_ref).copy_(inp_ref)
  res = torch.empty_like(res_ref).copy_(res_ref)
  inp_flashinfer = torch.empty_like(inp_ref).copy_(inp_ref)
  res_flashinfer = torch.empty_like(res_ref).copy_(res_ref)

  ref_out, inp_res = ref_rms_norm(inp_ref, res_ref, weight, eps)
  fused_add_rmsnorm(inp_flashinfer, res_flashinfer, weight, eps=eps)
  hp_rms_norm.hp_rms_norm(inp, weight, res, eps)

  torch.cuda.synchronize()
  torch.testing.assert_close(inp, ref_out, rtol=rtol, atol=atol)
  torch.testing.assert_close(res, inp_res, rtol=rtol, atol=atol)
  torch.testing.assert_close(inp, inp_flashinfer, rtol=rtol, atol=atol)
  torch.testing.assert_close(res, res_flashinfer, rtol=rtol, atol=atol)

if __name__ == '__main__':
  cc_major = torch.cuda.get_device_capability()[0]
  if cc_major == 9:
    upper_bound = 24448
  elif cc_major == 10:
    upper_bound = 32768

  for dtype in [torch.bfloat16, torch.float16]:
    for hidden_dim in range(128, upper_bound + 128, 128):
      tokens = random.randint(1, 4096)
      check_diff(tokens, hidden_dim, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
      print(f"Check tokens {tokens}, hidden_dim {hidden_dim}, dtype {dtype} Done!")

  # for h in range(1024, 16384 + 1024, 1024):
  #   check_diff(4096, h, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
  #   print(f"hidden_dim {h}, Done!")
  # if cc_major == 10:
  #   for h in range(16384 + 4096, 32768 + 1024, 4096):
  #     check_diff(4096, h, eps=torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16)
  #     print(f"hidden_dim {h}, Done!")