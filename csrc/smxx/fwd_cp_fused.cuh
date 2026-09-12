#pragma once

#include "fwd_kernel1.cuh"
#include "fwd_state_only.cuh"

// M11-F0:
// Fused CP warmup preparation + state recurrence.
// Diagnostic specialization for Mixed H96 CP-P2.
//
// grid = (2, 96)
// block = 256
//
// compact block 0 -> CP segment 2, warmup 5 chunks
// compact block 1 -> CP segment 6, warmup 4 chunks
//
// Output remains FP32 ht_buffer for the admission test.
template <int D, int CHUNK = 16>
struct CpFusedLayouts {
    using K1L = K1Layouts<D, CHUNK>;
    using SOL = StateOnlyLayouts<D, CHUNK>;

    using BF16 = cutlass::bfloat16_t;

    using QKLayout = typename K1L::QKLayout;
    using GLayout = typename K1L::GLayout;
    using BetaSmemLayout = typename K1L::BetaSmemLayout;
    using GTotalLayout = typename K1L::GTotalLayout;
    using LMLayout = typename K1L::LMLayout;
    using MMALayout = typename K1L::MMALayout;

    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;

    using VOLayout = typename SOL::VOLayout;
    using TMAVOLayout = typename SOL::TMAVOLayout;

    using StateSmemLayout = typename SOL::StateSmemLayout;
    using TransposedStateSmemLayout =
        typename SOL::TransposedStateSmemLayout;
    using TransposedMMALayout =
        typename SOL::TransposedMMALayout;

    using FP32StateSmemLayout =
        typename SOL::FP32StateSmemLayout;
    using TMAFP32StateSmemLayout =
        typename SOL::TMAFP32StateSmemLayout;
};


template <class Layouts>
struct SharedStorageCpFused {
    using BF16 = cutlass::bfloat16_t;

    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;

    // Persistent across all 4/5 warmup chunks.
    alignas(128)
    cute::ArrayEngine<
        BF16,
        cute::cosize_v<StateSmemLayout>
    > state_acc;

        // ------------------------------------------------------------
    // Temporary storage after state_acc.
    //
    // Chunk scratch is alive only while processing warmup chunks.
    // state_fp32_buf is alive only after all chunks finish.
    //
    // Therefore they can share the same shared-memory region.
    // ------------------------------------------------------------

    struct ChunkScratch {
        // Phase A:
        //   k + g
        //
        // Phase B:
        //   k_decayed + k_inv + L + INV
        //
        // Their lifetimes do not overlap.
        union {
            struct {
                alignas(128)
                cute::ArrayEngine<
                    BF16,
                    cute::cosize_v<QKLayout>
                > k;

                alignas(128)
                cute::ArrayEngine<
                    float,
                    cute::cosize_v<GLayout>
                > g;
            };

            struct {
                alignas(128)
                cute::ArrayEngine<
                    BF16,
                    cute::cosize_v<MMALayout>
                > k_decayed;

                alignas(128)
                cute::ArrayEngine<
                    BF16,
                    cute::cosize_v<MMALayout>
                > k_inv;

                alignas(128)
                cute::ArrayEngine<
                    BF16,
                    cute::cosize_v<LMLayout>
                > L;

                alignas(128)
                cute::ArrayEngine<
                    BF16,
                    cute::cosize_v<LMLayout>
                > INV;
            };
        };

        // Still needed by state recurrence after preparation.
        alignas(128)
        cute::ArrayEngine<
            BF16,
            cute::cosize_v<MMALayout>
        > k_restored;

        // Raw inputs for the current chunk.
        alignas(128)
        cute::ArrayEngine<
            BF16,
            cute::cosize_v<QKLayout>
        > g_bf16;

        alignas(128)
        cute::ArrayEngine<
            BF16,
            cute::cosize_v<VOLayout>
        > v;

        alignas(128)
        cute::ArrayEngine<
            BF16,
            cute::cosize_v<BetaSmemLayout>
        > beta;

        union {
            alignas(128)
            cute::ArrayEngine<
                float,
                cute::cosize_v<GTotalLayout>
            > dt_bias;

            alignas(128)
            cute::ArrayEngine<
                float,
                cute::cosize_v<GTotalLayout>
            > g_total;
        };
    };

    union {
        ChunkScratch chunk;

        alignas(128)
        char state_fp32_buf[
            cute::cosize_v<StateSmemLayout> * sizeof(float)
        ];
    };

    alignas(16)
    cutlass::arch::ClusterTransactionBarrier
        tma_load_barrier;
};
