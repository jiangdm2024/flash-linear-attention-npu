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


#ifndef PREPARE_WY_REPR_BWD_FULL_VECTOR_H
#define PREPARE_WY_REPR_BWD_FULL_VECTOR_H


using namespace AscendC;

template <typename kType, typename betaType>
class PrepareWyReprBwdFullVectorProcess {
 public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessKBeta();
    __aicore__ inline void Init(GM_ADDR tiling, AscendC::TPipe *pipe_);
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
    AscendC::TPipe *pipe = nullptr;
private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> kBetaOutQue;
    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_)
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
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Init(GM_ADDR tiling, AscendC::TPipe *pipe_) {
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Process() {
    //计算K * Beta[:None]
    ProcessKBeta();
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBeta() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = BT;
    uint32_t rowOffset = 0;
    if(GetSubBlockNum() > 1) {
        if(GetSubBlockIdx() == 0) {
            rowNum = rowNum / GetSubBlockNum();
        } else {
            rowOffset = rowNum / GetSubBlockNum();
            rowNum = rowNum - rowOffset;
        }
    }
    //init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetaOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    auto tensorKfp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
            auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            //AscendC::printf("CrossCoreWaitFlag kOffset:%d, betaOffset:%d\n", kOffset, betaOffset);
            //copyin
            {
                auto tensorKin = kInQue.AllocTensor<kType>();
                DataCopy(tensorKin, kTensor[kOffset], K * rowNum);
                auto tensorBetain = betaInQue.AllocTensor<betaType>();
                DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);

                kInQue.EnQue(tensorKin);
                betaInQue.EnQue(tensorBetain);
                //AscendC::printf("copyin\n");
            }
            //compute
            {
                //AscendC::printf("150\n");
                auto tensorKin = kInQue.DeQue<kType>();
                auto tensorBetain = betaInQue.DeQue<betaType>();
                auto tensorOut = kBetaOutQue.AllocTensor<kType>();
                //AscendC::printf("153\n");
                //cast fp32
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                } else {
                    DataCopy(tensorBetafp32, tensorBetain, rowNum);
                }
                //AscendC::printf("159\n");
                Cast(tensorKfp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                PipeBarrier<PIPE_V>();
                //brcb
                Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                // DumpTensor(tensorBetaBrcbfp32, 0,  8 * rowNum);
                PipeBarrier<PIPE_V>();
                //AscendC::printf("165\n");
                //mul
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < K) {
                    Mul(tensorKfp32[perchannelResOffset], tensorKfp32[perchannelResOffset], tensorBetaBrcbfp32,
                        FP32_PER_REPEAT, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT;
                }
                // DumpTensor(tensorKfp32, 1,  K * rowNum);
                PipeBarrier<PIPE_V>();
                Cast(tensorOut, tensorKfp32, RoundMode::CAST_RINT, K * rowNum);

                kInQue.FreeTensor(tensorKin);
                betaInQue.FreeTensor(tensorBetain);
                kBetaOutQue.EnQue(tensorOut);
                //AscendC::printf("compute\n");
            }
            //copyout
            {
                auto tensorOut = kBetaOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[kOffset], tensorOut, K * rowNum);
                kBetaOutQue.FreeTensor(tensorOut);
                // AscendC::printf("kOffset:%ld, K * rowNum:%d\n", kOffset, K * rowNum);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    // DumpTensor(workSpaceTensor, 0,  8192);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    //AscendC::printf("CrossCoreWaitFlag\n");
    //AscendC::printf("CrossCoreWaitFlag\n");
    return;
}



#endif  // PREPARE_WY_REPR_BWD_FULL_VECTOR_H
