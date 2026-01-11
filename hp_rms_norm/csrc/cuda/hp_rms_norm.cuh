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
  longlong4 memory_type;
  __nv_bfloat162 real_type[8];
};

union U32B_f162{
  longlong4 memory_type;
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
  using V = longlong4;
};

template<> struct UVTypeTrait<__half, 32> {
  using U = U32B_f162;
  using V = longlong4;
};

template<typename T>
__device__ __forceinline__ void square_sum(float2& acc, T& val);

template<>
__device__ __forceinline__ void square_sum<__nv_bfloat162>(float2& acc, __nv_bfloat162& val) {
  float2 valf = __bfloat1622float2(val);
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
__device__ __forceinline__ T rms_residual(T& val, T& weight, T&residual, float rsqrt_square_sum);

template<>
__device__ __forceinline__ __nv_bfloat162 rms_residual<__nv_bfloat162>(
    __nv_bfloat162& val,
    __nv_bfloat162& weight,
    __nv_bfloat162& residual,
    float rsqrt_square_sum) {
  float2 valf = __bfloat1622float2(val);
  float2 weightf = __bfloat1622float2(weight);
  float2 residualf = __bfloat1622float2(residual);
  return __float22bfloat162_rn(
    make_float2(
      valf.x * weightf.x * rsqrt_square_sum + residualf.x,
      valf.y * weightf.y * rsqrt_square_sum + residualf.y
    )
  );
}

template<>
__device__ __forceinline__ __half2 rms_residual<__half2>(
    __half2& val,
    __half2& weight,
    __half2& residual,
    float rsqrt_square_sum) {
  float2 valf = __half22float2(val);
  float2 weightf = __half22float2(weight);
  float2 residualf = __half22float2(residual);
  return __float22half2_rn(
    make_float2(
      valf.x * weightf.x * rsqrt_square_sum + residualf.x,
      valf.y * weightf.y * rsqrt_square_sum + residualf.y
    )
  );
}

template<typename T, int VEC_SIZE_IN_BYTE>
__global__ void rms_norm_vector_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ residual,
    T* __restrict__ output,
    int tokens,
    int vec_hidden_dim,
    float eps
) {
  using U = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::U;
  using V = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::V;

  extern __shared__ __align__(128) uint8_t shared_memory[];

  // Use for warp reduce
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

  // Used for TMA 1D G2S Copy Sync
  barrier* mbarrier = reinterpret_cast<barrier*>(shared_memory + vec_hidden_dim * VEC_SIZE_IN_BYTE + 32 * sizeof(float));
  
  int token_id = static_cast<int>(blockIdx.x);
  if (token_id < tokens) {
    // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
    //    b) Make initialized barrier visible in async proxy.
    if (threadIdx.x == 0) {
      init(mbarrier, blockDim.x);
      // cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);   // b)
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
          *mbarrier
      );
    }
  } else {
    return;
  }

  // 3b. All threads arrive on the barrier
  __syncthreads();
  barrier::arrival_token arrival_token = mbarrier->arrive();

  while (token_id < tokens) {
    U u;  // Save input
    U u_res;  // Save residual
    U u_weight; // Save weight
    U u_out;  // Save output
    float2 acc_square = make_float2(0.0f, 0.0f);

    // Load data
    if (threadIdx.x < vec_hidden_dim) {
      const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
      u.memory_type = p[threadIdx.x];
      const V* p_res = reinterpret_cast<const V*>(residual) + token_id * vec_hidden_dim;
      u_res.memory_type = p_res[threadIdx.x];
    
      if constexpr (VEC_SIZE_IN_BYTE == 16) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          for (int i = 0; i < 4; i++) {
            float2 val = __bfloat1622float2(u.real_type[i]);
            acc_square.x += val.x * val.x;
            acc_square.y += val.y * val.y;
          }
        } else if constexpr (std::is_same_v<T, __half>) {
          for (int i = 0; i < 4; i++) {
            float2 val = __half22float2(u.real_type[i]);
            acc_square.x += val.x * val.x;
            acc_square.y += val.y * val.y;
          }
        }
      } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          for (int i = 0; i < 8; i++) {
            float2 val = __bfloat1622float2(u.real_type[i]);
            acc_square.x += val.x * val.x;
            acc_square.y += val.y * val.y;
          }
        } else if constexpr (std::is_same_v<T, __half>) {
          for (int i = 0; i < 8; i++) {
            float2 val = __half22float2(u.real_type[i]);
            acc_square.x += val.x * val.x;
            acc_square.y += val.y * val.y;
          }
        }
      }
    }

    // Warp Reduce
    float warp_sum = cooperative_groups::reduce(
      cg_warp,
      acc_square.x + acc_square.y,
      cooperative_groups::plus<float>()
    );

    // Wirte warp_sum to buffer
    float* buffer = reinterpret_cast<float*>(shared_memory + vec_hidden_dim * VEC_SIZE_IN_BYTE);
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

      if (token_id == static_cast<int>(blockIdx.x)) {
        // First time
        mbarrier->wait(std::move(arrival_token));
      }

      const V* p_weight = reinterpret_cast<const V*>(shared_memory);
      u_weight.memory_type = p_weight[threadIdx.x];  // LDS

      if constexpr (VEC_SIZE_IN_BYTE == 16) {
        for (int i = 0; i < 4; i++) {
          u_out.real_type[i] = rms_residual(u.real_type[i], u_weight.real_type[i], u_res.real_type[i], rsqrt_square_sum);
        }
      } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
        for (int i = 0; i < 8; i++) {
          u_out.real_type[i] = rms_residual(u.real_type[i], u_weight.real_type[i], u_res.real_type[i], rsqrt_square_sum);
        }
      }

      V* p_out = reinterpret_cast<V*>(output) + token_id * vec_hidden_dim;
      p_out[threadIdx.x] = u_out.memory_type;
    }
    token_id += static_cast<int>(gridDim.x);
  }
}

template<typename T, int VEC_SIZE_IN_BYTE, int REG_VEC_SIZE>
__global__ void __maxnreg__(32) rms_norm_vector_kernel_plus(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ residual,
    T* __restrict__ output,
    int tokens,
    int vec_hidden_dim,
    float eps,
    int shm_round
) {
  constexpr int threads = 1024;
  constexpr int warps = 32;
  using U = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::U;
  using V = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::V;
  extern __shared__ __align__(128) uint8_t shared_memory[];

  // iteration
  int iteration = (vec_hidden_dim + threads - 1) / threads;

  // Use for warp reduce
  auto cg_block = cooperative_groups::this_thread_block();
  auto cta = cooperative_groups::tiled_partition<1024>(cg_block);

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
          shared_memory + shm_round * threads * VEC_SIZE_IN_BYTE,
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
    U u[REG_VEC_SIZE];  // Save input
    float2 acc_square = make_float2(0.0f, 0.0f);

    // Ptr offset
    const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
    #pragma unroll 1
    for (int i = 0; i < iteration; i++) {
      auto offset = threadIdx.x + i * threads;
      if (offset < vec_hidden_dim && i < REG_VEC_SIZE) {
        // Store in registers
        u[i].memory_type = p[offset];
        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          #pragma unroll 1
          for (int j = 0; j < 4; j++) {
            square_sum(acc_square, u[i].real_type[j]);
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          #pragma unroll 1
          for (int j = 0; j < 8; j++) {
            square_sum(acc_square, u[i].real_type[j]);
          }
        }
      } else if (offset < vec_hidden_dim) {
        // Store in shared memory
        V* p_shm = reinterpret_cast<V*>(shared_memory) + (i - REG_VEC_SIZE) * threads;
        U tmp;
        tmp.memory_type = p[offset];
        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          #pragma unroll 1
          for (int j = 0; j < 4; j++) {
            square_sum(acc_square, tmp.real_type[j]);
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          #pragma unroll 1
          for (int j = 0; j < 8; j++) {
            square_sum(acc_square, tmp.real_type[j]);
          }
        }
        p_shm[threadIdx.x] = tmp.memory_type;
      }
    }

    // CTA Reduce
    float cta_sum = cooperative_groups::reduce(
      cta,
      acc_square.x + acc_square.y,
      cooperative_groups::plus<float>()
    );
    float scale = rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(vec_hidden_dim * (VEC_SIZE_IN_BYTE / sizeof(T)))));

    if (token_id == static_cast<int>(blockIdx.x)) {
      // First time
      mbarrier.wait(std::move(arrival_token));
    }

    // TODO: RMS Norm
    const V* p_res = reinterpret_cast<const V*>(residual) + token_id * vec_hidden_dim;
    V* p_out = reinterpret_cast<V*>(output) + token_id * vec_hidden_dim;
    V* p_weight = reinterpret_cast<V*>(shared_memory + shm_round * threads * VEC_SIZE_IN_BYTE);
    #pragma unroll 1
    for (int i = 0; i < iteration; i++) {
      auto offset = threadIdx.x + i * threads;
      U u_out;
      U u_res;
      U u_weight;
      if (offset < vec_hidden_dim) {
        u_res.memory_type = p_res[offset];  // LDG
        u_weight.memory_type = p_weight[offset];  // LDS
      }
      if (offset < vec_hidden_dim && i < REG_VEC_SIZE) {
        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          #pragma unroll 1
          for (int j = 0; j < 4; j++) {
            u_out.real_type[j] = rms_residual(u[i].real_type[j], u_weight.real_type[j], u_res.real_type[j], scale);
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          #pragma unroll 1
          for (int j = 0; j < 8; j++) {
            u_out.real_type[j] = rms_residual(u[i].real_type[j], u_weight.real_type[j], u_res.real_type[j], scale);
          }
        }
        p_out[offset] = u_out.memory_type;
      } else if (offset < vec_hidden_dim) {
        V* p_shm = reinterpret_cast<V*>(shared_memory) + (i - REG_VEC_SIZE) * threads;
        U shm_inp;
        shm_inp.memory_type = p_shm[threadIdx.x];

        if constexpr (VEC_SIZE_IN_BYTE == 16) {
          #pragma unroll 1
          for (int j = 0; j < 4; j++) {
            u_out.real_type[j] = rms_residual(shm_inp.real_type[j], u_weight.real_type[j], u_res.real_type[j], scale);
          }
        } else if constexpr (VEC_SIZE_IN_BYTE == 32) {
          #pragma unroll 1
          for (int j = 0; j < 8; j++) {
            u_out.real_type[j] = rms_residual(shm_inp.real_type[j], u_weight.real_type[j], u_res.real_type[j], scale);
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
void launch_rms_norm_vector(
    T* output,
    const T* input,
    const T* weight,
    const T* residual,
    size_t tokens,
    int vec_hidden_dim,
    double eps,
    cudaStream_t stream
) {
  if (vec_hidden_dim <= 1024) {
    uint threads = (vec_hidden_dim + 31) / 32 * 32;

    dim3 block(threads, 1, 1);
    // 32 * sizeof(float) -- CTA Reduce
    // vec_hidden_dim * VEC_SIZE_IN_BYTE -- Save weight
    // 8 -- mbarrier
    int smem_size = 32 * sizeof(float) + vec_hidden_dim * VEC_SIZE_IN_BYTE + 8;
    void const* kernel_ptr = reinterpret_cast<void const*>(&rms_norm_vector_kernel<T, VEC_SIZE_IN_BYTE>);

    int max_active_blocks_per_sm = -1;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, kernel_ptr, vec_hidden_dim, smem_size));
    dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * max_active_blocks_per_sm, 1, 1);
  
    AT_CUDA_CHECK(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

    // Kernel Launch
    assert(tokens <= INT32_MAX);
    rms_norm_vector_kernel<T, VEC_SIZE_IN_BYTE><<<grid, block, smem_size, stream>>>(
      input, weight, residual, output, static_cast<int>(tokens), vec_hidden_dim, static_cast<float>(eps)
    );
  } else {
    constexpr int reg_vec_size = 2;
    dim3 block(1024, 1, 1);
    void const* kernel_ptr = reinterpret_cast<void const*>(&rms_norm_vector_kernel_plus<T, VEC_SIZE_IN_BYTE, reg_vec_size>);
    
    size_t smem_size = 0;
    AT_CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel_ptr, 2, 1024));
    assert(smem_size <= INT32_MAX);
    int shm_round = static_cast<int>((static_cast<int>(smem_size) - vec_hidden_dim * VEC_SIZE_IN_BYTE) / (1024 * VEC_SIZE_IN_BYTE));

    AT_CUDA_CHECK(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

    dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 2, 1, 1);

    // Kernel Launch
    assert(tokens <= INT32_MAX);
    assert((vec_hidden_dim + 1023 / 1024) <= (reg_vec_size + shm_round));

    rms_norm_vector_kernel_plus<T, VEC_SIZE_IN_BYTE, reg_vec_size><<<grid, block, smem_size, stream>>>(
      input, weight, residual, output, static_cast<int>(tokens), vec_hidden_dim, static_cast<float>(eps), shm_round
    );
  }
}

template<typename T>
void launch_rms_norm(
    T* output,
    const T* input,
    const T* weight,
    const T* residual,
    size_t tokens,
    int hidden_dim,
    double eps,
    cudaStream_t stream) {
  // TODO: B200
  constexpr int max_vec_size_byte = 16;
  int elements_in_vec = max_vec_size_byte / sizeof(T);
  assert(hidden_dim % elements_in_vec == 0);
  int vec_hidden_dim = hidden_dim / elements_in_vec;
  launch_rms_norm_vector<T, max_vec_size_byte>(
    output, input, weight, residual, tokens, vec_hidden_dim, eps, stream
  );
}

} // hp_rms_norm
