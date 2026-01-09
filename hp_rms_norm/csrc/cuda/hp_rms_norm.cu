#include <torch/all.h>

#include <tuple>

#include "hp_rms_norm.cuh"

void rms_norm(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& residual,
    double eps) {
  TORCH_CHECK(input.dim() >= 2, "input.dim() >= 2");
  TORCH_CHECK(input.dim() == output.dim(), "input.dim() == output.dim()");
  TORCH_CHECK(input.dim() == residual.dim(), "input.dim() == residual.dim()");
  int dim = static_cast<int>(input.dim());
  for (int i = 0; i < dim; i++) {
    TORCH_CHECK(input.size(i) == output.size(i), "input.size(i) == output.size(i)");
    TORCH_CHECK(input.size(i) == residual.size(i), "input.size(i) == residual.size(i)");
  }

  TORCH_CHECK(weight.dim() == 1, "weight.dim() == 1");
  auto hidden_dim = weight.size(0);
  TORCH_CHECK(input.size(-1) == hidden_dim, "input.size(-1) == hidden_dim");

  TORCH_CHECK(input.stride(-1) == 1, "input.stride(-1) == 1");
  TORCH_CHECK(output.stride(-1) == 1, "output.stride(-1) == 1");
  TORCH_CHECK(residual.stride(-1) == 1, "residual.stride(-1) == 1");

  auto tokens = input.view({-1, hidden_dim}).size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (input.dtype() == torch::kBFloat16) {
    hp_rms_norm::launch_rms_norm<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        static_cast<size_t>(tokens),
        static_cast<int>(hidden_dim),
        eps,
        stream);
  } else if (input.dtype() == torch::kFloat16) {
    hp_rms_norm::launch_rms_norm<__half>(
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        reinterpret_cast<const __half*>(residual.data_ptr()),
        static_cast<size_t>(tokens),
        static_cast<int>(hidden_dim),
        eps,
        stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
}
