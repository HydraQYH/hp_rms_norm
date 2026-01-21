#include <iostream>
#include <cassert>
#include <type_traits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cooperative_groups.h>

namespace hp_rms_norm {

__device__ __always_inline int4 ldg_prefetch(const int4 *ptr) {
  int4 ret;
  asm volatile ("ld.global.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr));
  return ret;
}

struct CollectiveVector {
  constexpr static int threads = 8;

#if __CUDACC_VER_MAJOR__ >= 13
  using storage_t = longlong4_32a;
#else
  using storage_t = longlong4;
#endif

  __device__ __always_inline static storage_t load(const storage_t* addr, int threadId) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
    static_assert(std::is_same_v<storage_t, longlong4_32a>);
    longlong4_32a* p = reinterpret_cast<longlong4_32a*>(addr);
    return p[threadId];
#else
    const int4* p = reinterpret_cast<const int4*>(addr);
    union {
      storage_t s;
      int4 _128b_arr[2];
    } U;
    U._128b_arr[0] = ldg_prefetch(&p[threadId]);
    U._128b_arr[1] = p[threadId + threads];
    return U.s;
#endif
  }

  __device__ __always_inline static int4 load_top(const storage_t* addr, int threadId) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
    static_assert(false);
#else
    const int4* p = reinterpret_cast<const int4*>(addr);
    return ldg_prefetch(&p[threadId]);
#endif
  }

  __device__ __always_inline static int4 load_bottom(const storage_t* addr, int threadId) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
    static_assert(false);
#else
    const int4* p = reinterpret_cast<const int4*>(addr);
    return p[threadId + threads];
#endif
  }

  __device__ __always_inline static void store(storage_t* addr, int threadId, storage_t& s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
    static_assert(std::is_same_v<storage_t, longlong4_32a>);
    longlong4_32a* p = reinterpret_cast<longlong4_32a*>(addr);
    p[threadId] = s;
#else
    union {
      storage_t s;
      int4 _128b_arr[2];
    } U;
    U.s = s;
    int4* p = reinterpret_cast<int4*>(addr);
    __stcs(&p[threadId], U._128b_arr[0]);
    __stcs(&p[threadId + threads], U._128b_arr[1]);
#endif
  }

  __device__ __always_inline static void store_top(storage_t* addr, int threadId, int4& s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
  static_assert(false);
#else
    int4* p = reinterpret_cast<int4*>(addr);
    __stcs(&p[threadId], s);
    
#endif
  }

  __device__ __always_inline static void store_bottom(storage_t* addr, int threadId, int4& s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDACC_VER_MAJOR__ >= 13)
  static_assert(false);
#else
    int4* p = reinterpret_cast<int4*>(addr);
    __stcs(&p[threadId + threads], s);
#endif
  }

};

template<typename scalar_t, bool loop>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    uint tokens,
    const int d
) {
  namespace cg = cooperative_groups;
  cg::grid_group g = cg::this_grid();
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<8> collective = cg::tiled_partition<8>(cta);
  using storage_t = typename CollectiveVector::storage_t;

  uint token_id = blockIdx.x;
  const scalar_t* p_input = input + token_id * d;
  scalar_t* p_out = out + token_id * d;
  auto collective_rank = collective.meta_group_rank();
  if constexpr (!loop) {
    const storage_t* _p_input = reinterpret_cast<const storage_t*>(reinterpret_cast<const char*>(p_input) + collective_rank * 256);
    storage_t* _p_out = reinterpret_cast<storage_t*>(reinterpret_cast<char*>(p_out) + collective_rank * 256);
    storage_t val = CollectiveVector::load(_p_input, collective.thread_rank());
    CollectiveVector::store(_p_out, collective.thread_rank(), val);
  } else {
    // 1024 threads -> 128 quarter warps
    constexpr int elements_per_iteration = 128 * (256 / sizeof(scalar_t));
    int iterations = d + (elements_per_iteration - 1) / elements_per_iteration; // cdiv
    for (int i = 0; i < iterations; i++) {
      if ((i * elements_per_iteration + collective_rank * (256 / sizeof(scalar_t))) < d) {
        const storage_t* _p_input = reinterpret_cast<const storage_t*>(
          reinterpret_cast<const char*>(p_input) + i * 128 * 256 + collective_rank * 256);
        storage_t* _p_out = reinterpret_cast<storage_t*>(
          reinterpret_cast<char*>(p_out) + i * 128 * 256 + collective_rank * 256);
        storage_t val = CollectiveVector::load(_p_input, collective.thread_rank());
        CollectiveVector::store(_p_out, collective.thread_rank(), val);
      }

    }
  }
}

template<typename scalar_t>
void launch_relu(
    const scalar_t* input,
    scalar_t* out,
    int64_t tokens,
    int dim,
    int sm_count,
    cudaStream_t stream
) {
  uint quarter_warp = static_cast<uint>(dim * sizeof(scalar_t) / 256);
  if (quarter_warp * 8 <= 1024) {
    dim3 block(quarter_warp * 8, 1, 1);
    dim3 grid(tokens, 1, 1);
    activation_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
      out, input, static_cast<uint>(tokens), dim
    );
  } else {
    dim3 block(1024, 1, 1);
    dim3 grid(tokens, 1, 1);
    activation_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
      out, input, static_cast<uint>(tokens), dim
    );
  }
}

} // namespace hp_rms_norm
