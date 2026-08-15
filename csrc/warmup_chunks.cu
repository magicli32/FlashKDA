#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <math_constants.h>

// get_warmup_chunks CUDA kernel
// Grid: (N,)  Block: (next power of two >= H), where H <= 256
// Each block processes one segment, each thread handles one head.
// Scans backwards from the end accumulating gate decay until all heads converge.
// Uses min over D dimensions (not mean) to guarantee the weakest dimension converges.
__global__ void get_warmup_chunks_kernel(
    const __nv_bfloat16* __restrict__ g_ptr,  // [T, H, D]
    const float* __restrict__ A_log,          // [H]
    const float* __restrict__ dt_bias,        // [H, D]
    float gate_scale,                         // lower_bound * log2(e)
    int chunk_size,
    float threshold,
    const int64_t* __restrict__ cu_seqlens,   // [N+1]
    int H, int D,
    int32_t* __restrict__ num_warmup,         // output [N]
    bool* __restrict__ fallback               // output [N]
) {
    int seg_idx = blockIdx.x;
    int h = threadIdx.x;  // head index

    // Padding threads must participate because the block uses __syncthreads().
    bool active = h < H;

    int64_t bos = cu_seqlens[seg_idx];
    int64_t eos = cu_seqlens[seg_idx + 1];
    int seg_len = (int)(eos - bos);
    int nc = (seg_len + chunk_size - 1) / chunk_size;

    float a_exp = active ? expf(A_log[h]) : 0.0f;

    // Per-head cumulative decay (negative, grows more negative)
    float g_cumsum = active ? 0.0f : -CUDART_INF_F;

    // Shared memory for cross-head reduction
    extern __shared__ float smem[];  // [H]

    int result_warmup = nc;
    bool found = false;

    for (int c = 0; c < nc && !found; c++) {
        int chunk_end = (int)(eos - c * chunk_size - 1);
        if (chunk_end < (int)bos) chunk_end = (int)bos;

        // Compute min over D: gate per-dim, take the weakest (least negative)
        float sig_min = 1.0f;  // sigmoid max is 1, start high
        if (active) {
            const __nv_bfloat16* g_row =
                g_ptr + (int64_t)chunk_end * H * D + (int64_t)h * D;
            for (int d = 0; d < D; d++) {
                float g_val = __bfloat162float(g_row[d]);
                float x = a_exp * (g_val + dt_bias[h * D + d]);
                float sig = 1.0f / (1.0f + expf(-x));
                sig_min = fminf(sig_min, sig);
            }
        }

        // gate_scale is negative, sig_min is the weakest sigmoid → least decay
        float decay = active ? gate_scale * sig_min * (float)chunk_size : 0.0f;
        g_cumsum += decay;

        // Reduction: find max g_cumsum across heads (max = least negative = slowest head)
        smem[h] = g_cumsum;
        __syncthreads();

        // Simple tree reduction for max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (h < stride) {
                smem[h] = fmaxf(smem[h], smem[h + stride]);
            }
            __syncthreads();
        }

        // Check convergence: all heads' cumulative decay exceeded threshold
        float max_val = smem[0];
        if (max_val < threshold) {
            result_warmup = c + 1;
            found = true;
        }
        __syncthreads();
    }

    // Write output (only head 0)
    if (h == 0) {
        num_warmup[seg_idx] = result_warmup;
        fallback[seg_idx] = !found;
    }
}

// Host wrapper
void get_warmup_chunks_cuda(
    torch::Tensor g,            // [1, T, H, D] bf16
    torch::Tensor A_log,        // [H] fp32
    torch::Tensor dt_bias,      // [H, D] fp32
    double lower_bound,
    torch::Tensor cu_seqlens,   // [N+1] int64
    int chunk_size,
    double threshold,
    torch::Tensor num_warmup,   // output [N] int32
    torch::Tensor fallback      // output [N] bool
) {
    int N = cu_seqlens.numel() - 1;
    int H = A_log.numel();
    int D = dt_bias.size(1);

    float gate_scale = (float)(lower_bound * 1.4426950408889634);

    auto g_flat = g.reshape({-1, H, D});  // [T, H, D]

    dim3 grid(N);
    int block_size = 1;
    while (block_size < H) block_size <<= 1;
    dim3 block(block_size);
    int smem_size = block_size * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    get_warmup_chunks_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_flat.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        gate_scale,
        chunk_size,
        (float)threshold,
        cu_seqlens.data_ptr<int64_t>(),
        H, D,
        num_warmup.data_ptr<int32_t>(),
        reinterpret_cast<bool*>(fallback.data_ptr<bool>())
    );
}
