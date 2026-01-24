/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_full.h
 * \brief
 */

#include "prepare_wy_repr_bwd_full_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef PREPARE_WY_REPR_BWD_FULL_CUBE_H
#define PREPARE_WY_REPR_BWD_FULL_CUBE_H


using namespace Catlass;

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmadBdk_,
    class BlockMmadBdkb_,
    class BlockMmadBdkbg_,
    class BlockMmadBkkT_,
    class BlockMmadBdvb_,
    class BlockScheduler_
>
class PrepareWyReprBwdFullTla {
public:
    using BlockMmadBdk = BlockMmadBdk_;
    using BlockMmadBdkb = BlockMmadBdkb_;
    using BlockMmadBdkbg = BlockMmadBdkbg_;
    using BlockMmadBkkT = BlockMmadBkkT_;
    using BlockMmadBdvb = BlockMmadBdvb_;
    using ArchTag = typename BlockMmadBdkb::ArchTag;
    using BdkL1TileShape = typename BlockMmadBdk::L1TileShape;
    using BdkbL1TileShape = typename BlockMmadBdkb::L1TileShape;
    using ElementDA = typename BlockMmadBdk::ElementA;
    using LayoutDA = typename BlockMmadBdk::LayoutA;
    using ElementKbeta = typename BlockMmadBdk::ElementB;
    using LayoutKbeta = typename BlockMmadBdk::LayoutB;
    using ElementDk = typename BlockMmadBdk::ElementC;
    using LayoutDk = typename BlockMmadBdk::LayoutC;

    // using ElementA = typename BlockMmadBdkb::ElementA;
    // using LayoutDA = typename BlockMmadBdkb::LayoutA;
    // using ElementK = typename BlockMmadBdkb::ElementB;
    using LayoutK = typename BlockMmadBdkb::LayoutB;
    // using ElementC = typename BlockMmadBdkb::ElementC;
    // using LayoutDkb = typename BlockMmadBdkb::LayoutC;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_BDK_M = tla::get<0>(BdkL1TileShape{});
    static constexpr uint32_t L1_TILE_BDK_N = tla::get<1>(BdkL1TileShape{});
    static constexpr uint32_t L1_TILE_BDK_K = tla::get<2>(BdkL1TileShape{});
    static constexpr uint32_t L1_TILE_BDKB_M = tla::get<0>(BdkbL1TileShape{});
    static constexpr uint32_t L1_TILE_BDKB_N = tla::get<1>(BdkbL1TileShape{});
    static constexpr uint32_t L1_TILE_BDKB_K = tla::get<2>(BdkbL1TileShape{});
    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrKbeta;
        LayoutKbeta layoutKbeta;
        GM_ADDR ptrDA;
        LayoutDA layoutDA;
        GM_ADDR ptrDk;
        LayoutDk layoutDk;
        GM_ADDR ptrK;
        LayoutK layoutK;
        uint64_t B = 1;
        uint64_t T = 32768;
        uint64_t H = 32;
        uint64_t K = 128;
        uint64_t V = 128;
        uint64_t BT = 64;
        uint64_t stage = 2;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrptrKbeta_, LayoutKbeta layoutKbeta_,GM_ADDR ptrDA_,LayoutDA layoutDA_,GM_ADDR ptrDk_,LayoutKbeta layoutDk_,GM_ADDR ptrK_,LayoutK layoutK_,
        uint64_t B_, uint64_t T_,uint64_t H_,uint64_t K_,uint64_t V_,uint64_t BT_, uint64_t stage_)
            : ptrKbeta(ptrptrKbeta_), 
            layoutKbeta(layoutKbeta_),
            ptrDA(ptrDA_),
            layoutDA(layoutDA_),
            ptrDk(ptrDk_), 
            layoutDk(layoutDk_),
            ptrK(ptrK_),
            layoutK(layoutK_),
            B(B_), 
            T(T_), 
            H(H_), 
            K(K_), 
            V(V_), 
            BT(BT_), 
            stage(stage_){}
    };

    // Methods
    CATLASS_DEVICE
    PrepareWyReprBwdFullTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        GemmCoord ProblemShapeDk{static_cast<uint32_t>(params.T),static_cast<uint32_t>(params.K), static_cast<uint32_t>(params.BT)}; 
        BlockScheduler matmulBlockSchedulerDk(ProblemShapeDk, MakeCoord(L1_TILE_BDK_M, L1_TILE_BDK_N));
        Arch::Resource<ArchTag> resource;
        {   //处理第一部分cube DA @ Kbeta     V->C
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
            //AscendC::printf("CrossCoreSetFlag\n");
            //AscendC::printf("CrossCoreSetFlag\n");
            BlockMmadBdk blockMmadBdk(resource);
            uint32_t coreLoopsInB = matmulBlockSchedulerDk.GetCoreLoops();
            uint32_t coreLoops = params.B * CeilDiv(params.T, params.BT);
            uint32_t coreIdx = AscendC::GetBlockIdx();
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                uint32_t bIdx = loopIdx / coreLoopsInB;
                uint32_t chunkIdx = loopIdx % coreLoopsInB;
                GemmCoord blockCoord = matmulBlockSchedulerDk.GetBlockCoord(chunkIdx);
                GemmCoord actualBlockShape = matmulBlockSchedulerDk.GetActualBlockShape(blockCoord);
                // AscendC::printf("blockCoord.m(%d)  blockCoord.n(%d)\n",blockCoord.m(), blockCoord.n());
                for (int h = 0; h < params.H; h++) {

                    // Represent the full gm
                    AscendC::GlobalTensor<ElementDA> gmDA;
                    gmDA.SetGlobalBuffer((__gm__ ElementDA *)params.ptrDA + ((bIdx * params.H + h) * params.T * params.BT));
                    AscendC::GlobalTensor<ElementKbeta> gmKbeta;
                    gmKbeta.SetGlobalBuffer((__gm__ ElementKbeta *)params.ptrKbeta + ((bIdx * params.H + h) * params.T * params.K));
                    AscendC::GlobalTensor<ElementDk> gmDk;
                    gmDk.SetGlobalBuffer((__gm__ ElementDk *)params.ptrDk + ((bIdx * params.H + h) * params.T * params.K));

                    // Represent the full tensors
                    auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
                    auto tensorKbeta = tla::MakeTensor(gmKbeta, params.layoutKbeta, Arch::PositionGM{});
                    auto tensorDk = tla::MakeTensor(gmDk, params.layoutDk, Arch::PositionGM{});

                    AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
                    // Make tiled views
                    auto tensorBlockDA = GetTile(tensorDA,
                                                tla::MakeCoord(blockCoord.m() * L1_TILE_BDK_M, blockCoord.k() * L1_TILE_BDKB_K),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockDk = GetTile(tensorDk,
                                                tla::MakeCoord(blockCoord.m() * L1_TILE_BDK_M, blockCoord.n() * L1_TILE_BDKB_N),
                                                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    // Compute block-scoped matrix multiply-add

                    auto tensorBlockKbeta = GetTile(tensorKbeta,
                                                tla::MakeCoord(blockCoord.k() * L1_TILE_BDK_K, blockCoord.n() * L1_TILE_BDKB_N),
                                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    //AscendC::printf("CrossCoreWaitFlag\n");
                    blockMmadBdk(tensorBlockDA, tensorBlockKbeta, tensorBlockDk, actualBlockShape);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
                    //AscendC::printf("CrossCoreSetFlag\n");
                }
            }
        }
        // BlockScheduler matmulBlockSchedulerDkb(ProblemShapeDkb, MakeCoord(L1_TILE_BDK_M, L1_TILE_BDK_N));
        // GemmCoord &ProblemShapeDkb = ProblemShapeDk;
        // uint32_t coreLoopsInB = CeilDiv(params.T, params.BT);
        // BlockMmadBdkb blockMmadBdkb(resource);
        // for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        //     uint32_t bIdx = loopIdx / coreLoopsInB;
        //     GemmCoord blockCoord = matmulBlockSchedulerDkb.GetBlockCoord(loopIdx);
        //     GemmCoord actualBlockShape = matmulBlockSchedulerDkb.GetActualBlockShape(blockCoord);
        //     for (int h = 0; h < params.H; h++) {
        //         // Represent the full gm
        //         AscendC::GlobalTensor<ElementA> gmDA;
        //         gmDA.SetGlobalBuffer((__gm__ ElementA *)params.ptrDA + ((bIdx * params.H + h) * params.T * params.BT));
        //         AscendC::GlobalTensor<ElementK> gmK;
        //         gmK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK + ((bIdx * params.H + h) * params.T * params.K));
        //         AscendC::GlobalTensor<ElementC> gmWorkspaceDkb;
        //         gmWorkspaceDkb.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace + ((bIdx * params.H + h) * params.T * params.K));

        //         // Represent the full tensors
        //         auto tensorDA = tla::MakeTensor(gmDA, params.layoutDA, Arch::PositionGM{});
        //         auto tensorK = tla::MakeTensor(gmK, params.layoutK, Arch::PositionGM{});
        //         auto tensorDkb = tla::MakeTensor(gmWorkspaceDkb, params.layoutDkb, Arch::PositionGM{});


        //         // Make tiled views
        //         auto tensorBlockDA = GetTile(tensorDA,
        //                                     tla::MakeCoord(blockCoord.m() * L1_TILE_BDKB_M, blockCoord.k() * L1_TILE_BDKB_K),
        //                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
        //         auto tensorBlockK = GetTile(tensorK,
        //                                     tla::MakeCoord(blockCoord.k() * L1_TILE_BDKB_K, blockCoord.n() * L1_TILE_BDKB_N),
        //                                     tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
        //         auto tensorBlockDkb = GetTile(tensorDkb,
        //                                     tla::MakeCoord(blockCoord.m() * L1_TILE_BDKB_M, blockCoord.n() * L1_TILE_BDKB_N),
        //                                     tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

        //         // Compute block-scoped matrix multiply-add
        //         AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_3);
        //         blockMmadBdkb(tensorBlockDA, tensorBlockK, tensorBlockDkb, actualBlockShape);
        //         AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_5);
        //     }
        // }
    }

};
}

template <typename kType, typename betaType>
class PrepareWyReprBwdFullProcess {
 public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_);

    __aicore__ inline void Process();

    __aicore__ inline void Init(GM_ADDR tiling);
private:
    uint64_t B = 1;
    uint64_t T = 2048;
    uint64_t H = 4;
    uint64_t K = 128;
    uint64_t V = 128;
    uint64_t BT = 64;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwdFullProcess<kType, betaType>::PrepareWyReprBwdFullProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_)
 :
    k(k_),
    v(v_),
    beta(beta_),
    A(A_),
    dA(dA_),
    dw(dw_),
    du(du_),
    g(g_),
    dk(dk_),
    dv(dv_),
    dbeta(dbeta_),
    dg(dg_),
    workspace(workspace_)
    {};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Init(GM_ADDR tiling) {
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullProcess<kType, betaType>::Process() {

    //输入
    using LayoutTagA = layout::RowMajor;
    using LayoutTagDW = layout::RowMajor;
    using LayoutTagDA = layout::RowMajor;
    using LayoutTagDAT = layout::ColumnMajor;
    using LayoutTagBeta = layout::RowMajor;
    using LayoutTagK = layout::RowMajor;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagKT = layout::ColumnMajor;


    //输入
    LayoutTagA tagA = LayoutTagA::MakeLayout<kType>(T, BT);
    LayoutTagDW tagDW = LayoutTagDW::MakeLayout<kType>(T, K);
    LayoutTagDA tagDA = LayoutTagDA::MakeLayout<kType>(T, BT);
    LayoutTagDAT tagDAT = LayoutTagDAT::MakeLayout<kType>(T, BT);
    LayoutTagK tagK = LayoutTagK::MakeLayout<kType>(T, K);
    LayoutTagV tagV = LayoutTagDW::MakeLayout<kType>(T, V);
    LayoutTagKT tagKT = LayoutTagKT::MakeLayout<kType>(T, K);

    //中间结果
    using LayoutTagKbeta = layout::RowMajor;
    LayoutTagKbeta tagKbeta = LayoutTagKbeta::MakeLayout<kType>(T, K);

    using LayoutTagDkb = layout::RowMajor;
    LayoutTagDkb tagDkb = LayoutTagDkb::MakeLayout<kType>(T, K);

    using LayoutTagDkbg = layout::RowMajor;
    LayoutTagV tagDkbg = LayoutTagDkbg::MakeLayout<kType>(T, V);

    //输出
    using LayoutTagDk = layout::RowMajor;
    LayoutTagDk tagDk = LayoutTagDk::MakeLayout<kType>(T, K);

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = Shape<_64, _256, _256>;
    using L0TileShape = Shape<_64, _256, _64>;

    //计算dk第一部分, dA @ Kbeta
    using TileCopyDk =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagKbeta, kType, LayoutTagK>;
    using BlockMmadDk = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDk>;

    using TileCopyDkb =
        Gemm::Tile::PackedTileCopyTla<ArchTag, kType, LayoutTagDA, kType, LayoutTagK, kType, LayoutTagDkbg>;
    using BlockMmadDkb = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, kType, kType, kType, void, TileCopyDkb>;
    // using CType = Gemm::GemmType<half, LayoutTagDkb>;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    auto layoutKbeta = MakeLayoutFromTag(tagKbeta);
    auto layoutDA = MakeLayoutFromTag(tagDA);
    auto layoutDK = MakeLayoutFromTag(tagDk);
    // auto layoutDAT = MakeLayoutFromTag(tagDAT);
    auto layoutK = MakeLayoutFromTag(tagK);

    // kernel level
    using MatmulKernel = Gemm::Kernel::PrepareWyReprBwdFullTla<BlockMmadDk, BlockMmadDkb, BlockMmadDkb, BlockMmadDkb, BlockMmadDkb, BlockScheduler>;

    MatmulKernel kernel;

    typename MatmulKernel::Params param{workspace, layoutKbeta,dA, layoutDA,dk, layoutDK, k, layoutK, B, T, H, K, V, BT, 4};
    kernel(param);
}


#endif  // PREPARE_WY_REPR_BWD_FULL_CUBE_H
