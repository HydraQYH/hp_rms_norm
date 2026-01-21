#include <iostream>
#include <cassert>
#include <type_traits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <torch/all.h>

namespace hp_rms_norm {

using barrier = cuda::barrier<cuda::thread_scope_block>;
// namespace ptx = cuda::ptx;

union U16B_bf162{
  int4 memory_type;
  __nv_bfloat162 real_type[4];
};

union U16B_f162{
  int4 memory_type;
  __half2 real_type[4];
};

union U32B_bf162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __nv_bfloat162 real_type[8];
};

union U32B_f162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __half2 real_type[8];
};

template<typename T, int VEC_SIZE_IN_BYTE>
struct UVTypeTrait;

template<> struct UVTypeTrait<__nv_bfloat16, 16> {
  using U = U16B_bf162;
  using V = int4;
};

template<> struct UVTypeTrait<__half, 16> {
  using U = U16B_f162;
  using V = int4;
};

template<> struct UVTypeTrait<__nv_bfloat16, 32> {
  using U = U32B_bf162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};

template<> struct UVTypeTrait<__half, 32> {
  using U = U32B_f162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};

template<typename T>
__device__ __forceinline__ void square_sum(float2& acc, T& val);

template<>
__device__ __forceinline__ void square_sum<__nv_bfloat162>(float2& acc, __nv_bfloat162& val) {
  float2 valf = __bfloat1622float2(val);
  float2 resf = __bfloat1622float2(val);
  acc.x += valf.x * valf.x;
  acc.y += valf.y * valf.y;
}

template<>
__device__ __forceinline__ void square_sum<__half2>(float2& acc, __half2& val) {
  float2 valf = __half22float2(val);
  acc.x += valf.x * valf.x;
  acc.y += valf.y * valf.y;
}

template<typename T>
__device__ __forceinline__ T rms(T& val, T& weight, float rsqrt_square_sum);

template<>
__device__ __forceinline__ __nv_bfloat162 rms<__nv_bfloat162>(
    __nv_bfloat162& val,
    __nv_bfloat162& weight,
    float rsqrt_square_sum) {
  float2 valf = __bfloat1622float2(val);
  float2 weightf = __bfloat1622float2(weight);
  return __float22bfloat162_rn(
    make_float2(
      valf.x * weightf.x * rsqrt_square_sum,
      valf.y * weightf.y * rsqrt_square_sum
    )
  );
}

template<>
__device__ __forceinline__ __half2 rms<__half2>(
    __half2& val,
    __half2& weight,
    float rsqrt_square_sum) {
  float2 valf = __half22float2(val);
  float2 weightf = __half22float2(weight);
  return __float22half2_rn(
    make_float2(
      valf.x * weightf.x * rsqrt_square_sum,
      valf.y * weightf.y * rsqrt_square_sum
    )
  );
}

template<typename T, int VEC_SIZE_IN_BYTE>
__global__ void rms_norm_vector_reg_kernel(
    T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ residual,
    int tokens,
    int vec_hidden_dim,
    float eps
) {
  using U = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::U;
  using V = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::V;
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

  __shared__ float shared_memory[32];

  U u;  // Save input
  U u_weight; // Save weight
  U u_out;  // Save output
  float2 acc_square = make_float2(0.0f, 0.0f);

  int token_id = static_cast<int>(blockIdx.x);

  // Load data
  if (threadIdx.x < vec_hidden_dim) {
    U u_res;  // Save residual
    const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
    u.memory_type = p[threadIdx.x];
    V* p_res = reinterpret_cast<V*>(residual) + token_id * vec_hidden_dim;
    u_res.memory_type = p_res[threadIdx.x];
    const V* p_weight = reinterpret_cast<const V*>(weight);
    if constexpr (std::is_same_v<V, int4>) {
      u_weight.memory_type = __ldg(&p_weight[threadIdx.x]);
    } else {
      u_weight.memory_type = p_weight[threadIdx.x];
    }

    if constexpr (VEC_SIZE_IN_BYTE == 16) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        for (int i = 0; i < 4; i++) {
          float2 val = __bfloat1622float2(u.real_type[i]);
          float2 res = __bfloat1622float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22bfloat162_rn(inp_res);
        }
      } else if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < 4; i++) {
          float2 val = __half22float2(u.real_type[i]);
          float2 res = __half22float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22half2_rn(inp_res);
        }
      }
    } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        for (int i = 0; i < 8; i++) {
          float2 val = __bfloat1622float2(u.real_type[i]);
          float2 res = __bfloat1622float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22bfloat162_rn(inp_res);
        }
      } else if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < 8; i++) {
          float2 val = __half22float2(u.real_type[i]);
          float2 res = __half22float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22half2_rn(inp_res);
        }
      }
    }

    p_res[threadIdx.x] = u.memory_type;
  }

  // Warp Reduce
  float warp_sum = cooperative_groups::reduce(
    cg_warp,
    acc_square.x + acc_square.y,
    cooperative_groups::plus<float>()
  );

  // Wirte warp_sum to buffer
  float* buffer = shared_memory;
  if (threadIdx.x % 32 == 0) {
    buffer[threadIdx.x / 32] = warp_sum;
  }

  // CTA Reduce
  __syncthreads();
  if (threadIdx.x < 32) {
    float cta_sum = cooperative_groups::reduce(
      cg_warp,
      (threadIdx.x < blockDim.x / 32) ? buffer[threadIdx.x] : 0.0f,
      cooperative_groups::plus<float>()
    );
    buffer[threadIdx.x] = rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(vec_hidden_dim * (VEC_SIZE_IN_BYTE / sizeof(T)))));
  }
  __syncthreads();

  // RMS Norm
  if (threadIdx.x < vec_hidden_dim) {
    // Read rsqrt from Shared Memory(Broadcast)
    float rsqrt_square_sum = buffer[threadIdx.x / 32];

    if constexpr (VEC_SIZE_IN_BYTE == 16) {
      for (int i = 0; i < 4; i++) {
        u_out.real_type[i] = rms(u.real_type[i], u_weight.real_type[i], rsqrt_square_sum);
      }
    } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
      for (int i = 0; i < 8; i++) {
        u_out.real_type[i] = rms(u.real_type[i], u_weight.real_type[i], rsqrt_square_sum);
      }
    }

    V* p_out = reinterpret_cast<V*>(input) + token_id * vec_hidden_dim;
    p_out[threadIdx.x] = u_out.memory_type;
  }
}

template<typename T, int VEC_SIZE_IN_BYTE, int NUM_WARPS, int MAXNREG>
__global__ void __maxnreg__(MAXNREG) rms_norm_vector_reg_shm_kernel(
    T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ residual,
    int tokens,
    int vec_hidden_dim,
    float eps,
    int shm_for_inp
) {
  constexpr int threads = NUM_WARPS * 32;
  using U = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::U;
  using V = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::V;
  extern __shared__ __align__(128) uint8_t shared_memory[];

  // iteration
  int iteration = (vec_hidden_dim + threads - 1) / threads; // iteration > 1

  // Use for warp reduce
  auto cg_block = cooperative_groups::this_thread_block();
  auto cg_warp = cooperative_groups::tiled_partition<32>(cg_block);

  int token_id = static_cast<int>(blockIdx.x);

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier mbarrier;

  if (token_id < tokens) {
    // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
    //    b) Make initialized barrier visible in async proxy.
    if (threadIdx.x == 0) {
      init(&mbarrier, blockDim.x);
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);   // b)
    }
    __syncthreads();

    // 2. Initiate TMA transfer to copy global to shared memory.
    if (threadIdx.x == 0) {
      // 3a. cuda::memcpy_async arrives on the barrier and communicates
      //     how many bytes are expected to come in (the transaction count)
      cuda::memcpy_async(
          shared_memory,
          weight,
          cuda::aligned_size_t<16>(vec_hidden_dim * VEC_SIZE_IN_BYTE),
          mbarrier
      );
    }
  } else {
    return;
  }

  // 3b. All threads arrive on the barrier
  __syncthreads();
  barrier::arrival_token arrival_token = mbarrier.arrive();

  while (token_id < tokens) {
    register U u;  // Save input
    U u_res;
    float2 acc_square = make_float2(0.0f, 0.0f);

    // Ptr offset
    const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
    V* p_res = reinterpret_cast<V*>(residual) + token_id * vec_hidden_dim;
    u.memory_type = p[threadIdx.x];
    u_res.memory_type = p_res[threadIdx.x];
    if constexpr (VEC_SIZE_IN_BYTE == 16) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        for (int i = 0; i < 4; i++) {
          float2 val = __bfloat1622float2(u.real_type[i]);
          float2 res = __bfloat1622float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22bfloat162_rn(inp_res);
        }
      } else if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < 4; i++) {
          float2 val = __half22float2(u.real_type[i]);
          float2 res = __half22float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22half2_rn(inp_res);
        }
      }
    } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        for (int i = 0; i < 8; i++) {
          float2 val = __bfloat1622float2(u.real_type[i]);
          float2 res = __bfloat1622float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22bfloat162_rn(inp_res);
        }
      } else if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < 8; i++) {
          float2 val = __half22float2(u.real_type[i]);
          float2 res = __half22float2(u_res.real_type[i]);
          float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
          acc_square.x += inp_res.x * inp_res.x;
          acc_square.y += inp_res.y * inp_res.y;
          u.real_type[i] = __float22half2_rn(inp_res);
        }
      }
    }
    p_res[threadIdx.x] = u.memory_type;

    V* p_shm = reinterpret_cast<V*>(shared_memory + VEC_SIZE_IN_BYTE * vec_hidden_dim);
    for (int i = 1; i < iteration; i++) {
      auto offset = threadIdx.x + i * threads;
      if (offset < vec_hidden_dim) {
        // Store in shared memory
        U tmp;
        tmp.memory_type = p[offset];
        u_res.memory_type = p_res[offset];
        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            for (int j = 0; j < 4; j++) {
              float2 val = __bfloat1622float2(tmp.real_type[j]);
              float2 res = __bfloat1622float2(u_res.real_type[j]);
              float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
              acc_square.x += inp_res.x * inp_res.x;
              acc_square.y += inp_res.y * inp_res.y;
              tmp.real_type[j] = __float22bfloat162_rn(inp_res);
            }
          } else if constexpr (std::is_same_v<T, __half>) {
            for (int j = 0; j < 4; j++) {
              float2 val = __half22float2(tmp.real_type[j]);
              float2 res = __half22float2(u_res.real_type[j]);
              float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
              acc_square.x += inp_res.x * inp_res.x;
              acc_square.y += inp_res.y * inp_res.y;
              tmp.real_type[j] = __float22half2_rn(inp_res);
            }
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            for (int j = 0; j < 8; j++) {
              float2 val = __bfloat1622float2(tmp.real_type[j]);
              float2 res = __bfloat1622float2(u_res.real_type[j]);
              float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
              acc_square.x += inp_res.x * inp_res.x;
              acc_square.y += inp_res.y * inp_res.y;
              tmp.real_type[j] = __float22bfloat162_rn(inp_res);
            }
          } else if constexpr (std::is_same_v<T, __half>) {
            for (int j = 0; j < 8; j++) {
              float2 val = __half22float2(tmp.real_type[j]);
              float2 res = __half22float2(u_res.real_type[j]);
              float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
              acc_square.x += inp_res.x * inp_res.x;
              acc_square.y += inp_res.y * inp_res.y;
              tmp.real_type[j] = __float22half2_rn(inp_res);
            }
          }
        }
        auto shm_offset = threadIdx.x + (i - 1) * threads;
        p_shm[shm_offset] = tmp.memory_type;
        p_res[offset] = tmp.memory_type;
      }
    }

    // Warp Reduce
    float warp_sum = cooperative_groups::reduce(
      cg_warp,
      acc_square.x + acc_square.y,
      cooperative_groups::plus<float>()
    );

    // Wirte warp_sum to buffer
    float* buffer = reinterpret_cast<float*>(shared_memory + vec_hidden_dim * VEC_SIZE_IN_BYTE + shm_for_inp );
    if (threadIdx.x % 32 == 0) {
      buffer[threadIdx.x / 32] = warp_sum;
    }

    // CTA Reduce
    __syncthreads();
    if (threadIdx.x < 32) {
      float cta_sum = cooperative_groups::reduce(
        cg_warp,
        threadIdx.x < NUM_WARPS ? buffer[threadIdx.x] : 0.0f,
        cooperative_groups::plus<float>()
      );
      buffer[threadIdx.x] = rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(vec_hidden_dim * (VEC_SIZE_IN_BYTE / sizeof(T)))));
    }
    __syncthreads();
    float scale = buffer[threadIdx.x / 32];

    if (token_id == static_cast<int>(blockIdx.x)) {
      // First time
      mbarrier.wait(std::move(arrival_token));
    }

    V* p_out = reinterpret_cast<V*>(input) + token_id * vec_hidden_dim;
    V* p_weight = reinterpret_cast<V*>(shared_memory);

    U u_out;
    U u_weight;
    u_weight.memory_type = p_weight[threadIdx.x];  // LDS
    if constexpr (VEC_SIZE_IN_BYTE == 16) {
      // #pragma unroll 1
      for (int j = 0; j < 4; j++) {
        u_out.real_type[j] = rms(u.real_type[j], u_weight.real_type[j], scale);
      }
    } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
      // #pragma unroll 1
      for (int j = 0; j < 8; j++) {
        u_out.real_type[j] = rms(u.real_type[j], u_weight.real_type[j], scale);
      }
    }
    p_out[threadIdx.x] = u_out.memory_type;

    for (int i = 1; i < iteration; i++) {
      auto offset = threadIdx.x + i * threads;
      if (offset < vec_hidden_dim) {
        u_weight.memory_type = p_weight[offset];  // LDS
        U shm_inp;
        auto shm_offset = threadIdx.x + (i - 1) * threads;
        shm_inp.memory_type = p_shm[shm_offset];

        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          // #pragma unroll 1
          for (int j = 0; j < 4; j++) {
            u_out.real_type[j] = rms(shm_inp.real_type[j], u_weight.real_type[j], scale);
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          // #pragma unroll 1
          for (int j = 0; j < 8; j++) {
            u_out.real_type[j] = rms(shm_inp.real_type[j], u_weight.real_type[j], scale);
          }
        }
        p_out[offset] = u_out.memory_type;
      }
    }

    // Move to next token
    token_id += static_cast<int>(gridDim.x);
  }
}

template<typename T, int VEC_SIZE_IN_BYTE>
void launch_rms_norm_vector_reg(
    T* input,
    const T* weight,
    T* residual,
    size_t tokens,
    int vec_hidden_dim,
    double eps,
    cudaStream_t stream
) {
  // Align to 32(warp)
  uint threads = (vec_hidden_dim + 31) / 32 * 32;
  dim3 block(threads, 1, 1);
  dim3 grid(static_cast<uint>(tokens), 1, 1);

  // Kernel Launch
  TORCH_CHECK(tokens <= INT32_MAX, "tokens <= INT32_MAX");
  rms_norm_vector_reg_kernel<T, VEC_SIZE_IN_BYTE><<<grid, block, 0, stream>>>(
    input, weight, residual, static_cast<int>(tokens), vec_hidden_dim, static_cast<float>(eps)
  );
}

template<typename T, int VEC_SIZE_IN_BYTE, int NUM_THREADS, int NUM_CTAS_PER_SM>
void launch_rms_norm_vector_reg_shm(
    T* input,
    const T* weight,
    T* residual,
    size_t tokens,
    int vec_hidden_dim,
    double eps,
    cudaStream_t stream
) {
  constexpr int num_threads = NUM_THREADS;
  constexpr int num_warps = NUM_THREADS / 32;
  constexpr int num_ctas_per_sm = NUM_CTAS_PER_SM;
  constexpr int maxnreg = (num_threads == 1024 ? 32 : 64);
  dim3 block(num_threads, 1, 1);
  void const* kernel_ptr = reinterpret_cast<void const*>(&rms_norm_vector_reg_shm_kernel<T, VEC_SIZE_IN_BYTE, num_warps, maxnreg>);

  cudaFuncAttributes kernel_attr;
  AT_CUDA_CHECK(cudaFuncGetAttributes(&kernel_attr, kernel_ptr));
  AT_CUDA_CHECK(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin - kernel_attr.sharedSizeBytes));

  size_t smem_size = 0;
  int cuda_runtime_version = 0;
  AT_CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
  if (cuda_runtime_version >= 13000) {
    AT_CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel_ptr, num_ctas_per_sm, num_threads));
  } else {
    smem_size = 81920;
  }
  uint persistent_ctas = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * num_ctas_per_sm;
  int shm_for_inp = static_cast<int>(smem_size) - vec_hidden_dim * VEC_SIZE_IN_BYTE - 128;
  TORCH_CHECK(shm_for_inp >= (vec_hidden_dim * VEC_SIZE_IN_BYTE - num_threads * VEC_SIZE_IN_BYTE), "hidden_dim too large.");

  dim3 grid(persistent_ctas, 1, 1);
  // Kernel Launch
  TORCH_CHECK(tokens <= INT32_MAX, "tokens <= INT32_MAX");
  rms_norm_vector_reg_shm_kernel<T, VEC_SIZE_IN_BYTE, num_warps, maxnreg><<<grid, block, smem_size, stream>>>(
    input, weight, residual, static_cast<int>(tokens), vec_hidden_dim, static_cast<float>(eps), shm_for_inp
  );
}

template<typename T>
void launch_rms_norm(
    T* input,
    const T* weight,
    T* residual,
    size_t tokens,
    int hidden_dim,
    double eps,
    cudaStream_t stream) {
  auto cc_major = at::cuda::getCurrentDeviceProperties()->major;
  TORCH_CHECK(cc_major >= 9, "High Performance RMSNorm only support in cc >= 90.");
  if ((cc_major == 9 && hidden_dim <= 8192) || (cc_major == 10 && hidden_dim <= 16384)) {
    int max_vec_size_byte = 0;
    if (cc_major == 10) {
      max_vec_size_byte = 32;
    } else {
      max_vec_size_byte = 16;
    }
    int elements_in_vec = max_vec_size_byte / sizeof(T);
    TORCH_CHECK(hidden_dim % elements_in_vec == 0, "High Performance RMSNorm need hidden_dim align to max_vec_size");

    int vec_hidden_dim = hidden_dim / elements_in_vec;
    if (max_vec_size_byte == 32) {
      launch_rms_norm_vector_reg<T, 32>(
        input, weight, residual, tokens, vec_hidden_dim, eps, stream
      );
    } else if (max_vec_size_byte == 16) {
      launch_rms_norm_vector_reg<T, 16>(
        input, weight, residual, tokens, vec_hidden_dim, eps, stream
      );
    };
  } else {
    if (cc_major == 9) {
      constexpr int max_vec_size_byte = 16;
      int elements_in_vec = max_vec_size_byte / sizeof(T);
      int vec_hidden_dim = hidden_dim / elements_in_vec;
#if __CUDACC_VER_MAJOR__ >= 13
      launch_rms_norm_vector_reg_shm<T, max_vec_size_byte, 512, 2>(
        input, weight, residual, tokens, vec_hidden_dim, eps, stream
      );
#else
      launch_rms_norm_vector_reg_shm<T, max_vec_size_byte, 1024, 2>(
        input, weight, residual, tokens, vec_hidden_dim, eps, stream
      );
#endif
    } else if (cc_major == 10) {
      constexpr int max_vec_size_byte = 32;
      int elements_in_vec = max_vec_size_byte / sizeof(T);
      int vec_hidden_dim = hidden_dim / elements_in_vec;
      launch_rms_norm_vector_reg_shm<T, max_vec_size_byte, 512, 2>(
        input, weight, residual, tokens, vec_hidden_dim, eps, stream
      );
    } else {
      TORCH_CHECK(false, "Unreachable.")
    }
  }
}

} // hp_rms_norm
