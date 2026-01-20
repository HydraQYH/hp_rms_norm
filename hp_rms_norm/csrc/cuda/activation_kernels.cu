#include <torch/all.h>

#include <tuple>

#include "activation_kernels.cuh"

void relu(
    torch::Tensor& out,
    const torch::Tensor& input
  ) {
  TORCH_CHECK(input.dim() >= 2, "input.dim() >= 2");
  int dim = static_cast<int>(input.dim());
  for (int i = 0; i < dim; i++) {
    TORCH_CHECK(input.size(i) == out.size(i), "input.size(i) == out.size(i)");
  }
  auto hidden_dim = input.size(dim - 1);
  auto tokens = input.view({-1, hidden_dim}).size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (input.dtype() == torch::kBFloat16) {
    hp_rms_norm::launch_relu<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int64_t>(tokens),
        static_cast<int>(hidden_dim),
        static_cast<int>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount),
        stream);
  } else if (input.dtype() == torch::kFloat16) {
    hp_rms_norm::launch_relu<__half>(
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        static_cast<int64_t>(tokens),
        static_cast<int>(hidden_dim),
        static_cast<int>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount),
        stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
}
