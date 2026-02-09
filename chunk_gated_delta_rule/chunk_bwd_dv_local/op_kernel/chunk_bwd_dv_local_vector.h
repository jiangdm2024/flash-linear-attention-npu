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
 * \file chunk_bwd_dv_local_base.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_VECTOR_H
#define CHUNK_BWD_DV_LOCAL_VECTOR_H

#include "kernel_operator.h"

namespace GDN {

template <typename QKVT, typename GT, typename Strategy>
class ChunkBwdDvLocalVector {
private:
    Strategy strategy;

public:
    __aicore__ inline ChunkBwdDvLocalVector(const Strategy &s) : strategy(s)
    {
    }
    __aicore__ inline void Process();

    __aicore__ inline void Init(GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix, GM_ADDR cu_seqlens,
                                GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe = nullptr);

    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<uint8_t> triMatrixGm;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> kqTQueIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFactorTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kqFp32TBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> kqTQueOut;

    int64_t H;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t subBlockNum;
    int64_t subBlockIdx;
    int64_t coreIdx;
    float scale;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<QKVT> qkvPadParams{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<GT> gPadParams{false, 0, 0, 0};
};

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Init(
    GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
    GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    gGm.SetGlobalBuffer((__gm__ GT *)g);
    triMatrixGm.SetGlobalBuffer((__gm__ uint8_t *)upper_tri_matrix);
    cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
    chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    H = tilingData->h;
    T = tilingData->t;
    K = tilingData->k;
    V = tilingData->v;
    scale = tilingData->scale;
    coreLoops = tilingData->b * strategy.chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    subBlockNum = AscendC::GetSubBlockNum();
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum);
    subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());

    pipe_ = pipe;
    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(GT));
    pipe_->InitBuffer(kqTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(QKVT));
    pipe_->InitBuffer(kqTQueOut, BUFFER_NUM, strategy.chunkSize * sizeof(QKVT));
    pipe_->InitBuffer(kqFp32TBuf, strategy.chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(gFp32TBuf, strategy.chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(gFactorTBuf, strategy.chunkSize * SIZE_FLOAT);
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Process()
{
    int64_t vecTaskIdx = 0;
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        int64_t curBatchId = static_cast<int64_t>(loopIdx) / strategy.chunkNumForT;
        IndexResult indexResult = strategy.calculate(loopIdx);
        if (indexResult.chunkLen <= 0) {
            continue;
        }
        int64_t baseOffset = 0;
        int64_t rowOffset = 0;
        for (int hIndex = 0; hIndex < H; hIndex++) {
            ++vecTaskIdx;
            if (vecTaskIdx % subBlockNum != subBlockIdx) {
                AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
                continue;
            }
            AscendC::LocalTensor<float> gFactorLocalTensor = gFactorTBuf.template Get<float>();
            AscendC::Duplicate<float>(gFactorLocalTensor, float(0.0), strategy.chunkSize);

            AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.AllocTensor<GT>();
            copyParams.blockLen = indexResult.chunkLen * sizeof(GT);
            AscendC::DataCopyPad(gLocalTensor, gGm[curBatchId * H * T + hIndex * T + indexResult.curTokenId],
                                 copyParams, gPadParams);
            MTE2ToVSync();
            AscendC::LocalTensor<float> gFp32LocalTensor;
            if constexpr (std::is_same<GT, float32_t>()) {
                gFp32LocalTensor = gLocalTensor;
            } else {
                gFp32LocalTensor = gFp32TBuf.template Get<float>();
                AscendC::Cast(gFp32LocalTensor, gLocalTensor, AscendC::RoundMode::CAST_NONE, indexResult.chunkLen);
            } 
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
            AscendC::LocalTensor<float> kqFp32LocalTensor = kqFp32TBuf.template Get<float>();
            AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.AllocTensor<QKVT>();
            AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.AllocTensor<QKVT>();
            
            copyParams.blockLen = indexResult.chunkLen * sizeof(QKVT);
            baseOffset = curBatchId * H * T * strategy.chunkSize + hIndex * T * strategy.chunkSize +
                                 indexResult.curTokenId * strategy.chunkSize;
            for (int row = 0; row < indexResult.chunkLen; row++) {
                rowOffset = baseOffset + row * strategy.chunkSize;
                AscendC::Adds(gFactorLocalTensor, gFp32LocalTensor, -1 * gFp32LocalTensor.GetValue(rowOffset),
                              indexResult.chunkLen);
                AscendC::Exp(gFactorLocalTensor, gFactorLocalTensor, indexResult.chunkLen);
                AscendC::Duplicate<float>(gFactorLocalTensor, float(0.0), row);
                AscendC::Muls(gFactorLocalTensor, gFactorLocalTensor, scale, indexResult.chunkLen);
                // 搬运 k * q^T 一行
                AscendC::DataCopyPad(
                    kqLocalTensor,
                    workspaceGm[rowOffset],
                    copyParams, qkvPadParams);
                MTE2ToVSync();
                AscendC::Cast(kqFp32LocalTensor, kqLocalTensor, AscendC::RoundMode::CAST_NONE, indexResult.chunkLen);
                AscendC::Mul(gFactorLocalTensor, kqFp32LocalTensor, gFactorLocalTensor, indexResult.chunkLen);
                AscendC::Cast(kqOutLocalTensor, gFactorLocalTensor, AscendC::RoundMode::CAST_NONE, strategy.chunkSize);
                // 搬出到workspace
                VToMTE3Sync();
                AscendC::DataCopy(workspaceGm[rowOffset], kqOutLocalTensor, strategy.chunkSize);
                MTE3ToVSync();
            }
            gTQueIn.FreeTensor(gLocalTensor);
            kqTQueIn.FreeTensor(kqLocalTensor);
            kqTQueOut.FreeTensor(kqOutLocalTensor);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
        }
    }

    AscendC::SyncAll<false>();
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_VECTOR_H