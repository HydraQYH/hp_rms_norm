import random
import torch
import hp_rms_norm

def check_diff(tokens: int, hidden_dim: int, dtype: torch.dtype=torch.bfloat16):
  if dtype == torch.bfloat16:
    rtol = 1.6e-2
    atol = 1e-3
  elif dtype == torch.float16:
    rtol = 1e-3
    atol = 1e-5

  inp_ref = torch.normal(0, 0.1, (tokens, hidden_dim), dtype=dtype, device='cuda')
  # out_ref = torch.zeros_like(inp_ref)

  inp = torch.empty_like(inp_ref).copy_(inp_ref)
  out = torch.zeros_like(inp)
  hp_rms_norm.relu(out, inp)

  torch.cuda.synchronize()

  # print(inp)
  # print(out)
  torch.testing.assert_close(inp, out)

if __name__ == '__main__':
  cc_major = torch.cuda.get_device_capability()[0]
  upper_bound = 32768
  for dtype in [torch.bfloat16, torch.float16]:
    for hidden_dim in range(128, upper_bound + 128, 128):
      tokens = random.randint(1, 4096)
      check_diff(tokens, hidden_dim, dtype=torch.bfloat16)
      print(f"Check tokens {tokens}, hidden_dim {hidden_dim}, dtype {dtype} Done!")
  # check_diff(4096, 8192, dtype=torch.bfloat16)
