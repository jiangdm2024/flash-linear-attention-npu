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
 * \file prepare_wy_repr_bwd_da_vector.h
 * \brief
 */

#ifndef PREPARE_WY_REPR_BWD_DA_VECTOR_H
#define PREPARE_WY_REPR_BWD_DA_VECTOR_H

using namespace AscendC;

template <typename kType, typename betaType>
class PrepareWyReprBwddAVectorProcess {
 public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwddAVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dA_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessKBeta();
    __aicore__ inline void ProcessDkb();
    __aicore__ inline void Init(GM_ADDR tiling, AscendC::TPipe *pipe_);
private:
    uint64_t B = 1;
    uint64_t T = 2048;
    uint64_t H = 4;
    uint64_t K = 128;
    uint64_t V = 128;
    uint64_t BT = 64;

    //这些会包在tiling data 里
    uint64_t dkRowNum = 32;
    uint64_t dkbRowNum = 32;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR dA;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;
private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    
    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> dwInQue;
    TQue<AscendC::TPosition::VECIN, 1> dwInQue;


    TQue<AscendC::TPosition::VECIN, 1> dkInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkbInQue;

    TQue<AscendC::TPosition::VECOUT, 1> kBetaGOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dkOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dBetaOutQue;

    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dkFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dkbFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> sharedTmpBuf;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwddAVectorProcess<kType, betaType>::PrepareWyReprBwddAVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dA_, GM_ADDR workspace_)
 :
    k(k_),
    v(v_),
    beta(beta_),
    A(A_),
    dw(dw_),
    du(du_),
    g(g_),
    dA(dA_),
    workspace(workspace_)
    {};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwddAVectorProcess<kType, betaType>::Init(GM_ADDR tiling, AscendC::TPipe *pipe_) {
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    // dkTensor.SetGlobalBuffer((__gm__ kType *)dk);
    // betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    // dbetaTensor.SetGlobalBuffer((__gm__ betaType *)dbeta);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwddAVectorProcess<kType, betaType>::Process() {
    // TODO 计算b_k_beta_g
    ProcessKBetaG();
    pipe->Reset();
    // TODO 计算b_v_beta
    // ProcessBVBeta();
    // pipe->Reset();
    // TODO 计算d_dA_tmp_1(b_dA_3) 和 d_dA_tmp_2(b_dA_4)
    // Process1KBetaG();
    // pipe->Reset();
    // TODO 计算d_dA_tmp_5(b_dA_7)
    // Process2KBetaG();
    // pipe->Reset();
    // 计算最终结果
    Process3KBetaG();

    //计算K * Beta[:None]
    ProcessKBeta();
    pipe->Reset();
    ProcessDkb();
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwddAVectorProcess<kType, betaType>::ProcessKBetaG() {
    // todo 如果此处是浅融合， 此处应该为第二个CV，需要等待第一个MM的结果
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
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(BetaFp32BrcbBuf, rowNum * ONE_BLOCK_32);

    //中间计算使用tmp
    pipe->InitBuffer(kBetaGOutQue, 2, rowNum * K * sizeof(kType));
    
    // 向外搬出的结果是workspace
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetaFp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFp32 = betaFp32BrcbBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
            auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
            auto gOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
            // AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            //AscendC::printf("CrossCoreWaitFlag kOffset:%d, betaOffset:%d\n", kOffset, betaOffset);
            //copyin
            {
                auto tensorKIn = kInQue.AllocTensor<kType>();
                DataCopy(tensorKIn, kTensor[kOffset], K * rowNum);
                auto tensorBetaIn = betaInQue.AllocTensor<betaType>();
                DataCopy(tensorBetaIn, betaTensor[betaOffset], rowNum);
                auto tensorGIn = gInQue.AllocTensor<betaType>();
                DataCopy(tensorGIn, gTensor[gOffset], rowNum);

                kInQue.EnQue(tensorKin);
                betaInQue.EnQue(tensorBetain);
                gInQue.EnQue(tensorGIn);
                //AscendC::printf("copyin\n");
            }
            //compute
            {
                //AscendC::printf("150\n");
                auto tensorKIn = kInQue.DeQue<kType>();
                auto tensorBetaIn = betaInQue.DeQue<betaType>();
                auto tensorGIn = gInQue.DeQue<betaType>();

                auto tensorOut = kBetaGOutQue.AllocTensor<kType>();
                //AscendC::printf("153\n");
                //cast fp32
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorBetafp32, tensorBetaIn, RoundMode::CAST_NONE, rowNum);
                } else {
                    DataCopy(tensorBetafp32, tensorBetaIn, rowNum);
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
                        FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                // DumpTensor(tensorKfp32, 1,  K * rowNum);
                // b_g处理
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorBetafp32, tensorGIn, RoundMode::CAST_NONE, rowNum);
                } else {
                    DataCopy(tensorBetafp32, tensorGIn, rowNum);
                }
                PipeBarrier<PIPE_V>();
                //brcb
                Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                PipeBarrier<PIPE_V>();
                // mul
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < K) {
                    Mul(tensorKfp32[perchannelResOffset], tensorKfp32[perchannelResOffset], tensorBetaBrcbfp32,
                        FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }

                // 输出
                PipeBarrier<PIPE_V>();
                Cast(tensorOut, tensorKfp32, RoundMode::CAST_RINT, K * rowNum);
                kInQue.FreeTensor(tensorKIn);
                betaInQue.FreeTensor(tensorBetaIn);
                gInQue.FreeTensor(tensorGIn);
                kBetaGOutQue.EnQue(tensorOut);
                //AscendC::printf("compute\n");
            }
            //copyout
            {
                auto tensorOut = kBetaGOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[kOffset], tensorOut, K * rowNum);
                kBetaGOutQue.FreeTensor(tensorOut);
            }
            // AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    // DumpTensor(workSpaceTensor, 0,  8192);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessVBeta() {
    uint32_t coreLoopsInB = CeilDiv(T, chunkSize);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = chunkSize;
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
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            auto kOffset = ((bIdx * H + h) * T  + chunkIdx * chunkSize + rowOffset) * K;
            auto betaOffset = (bIdx * H + h) * T  + chunkIdx * chunkSize + rowOffset;
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
                //cast FP32
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                } else {
                    DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                }
                //AscendC::printf("159\n");
                Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                PipeBarrier<PIPE_V>();
                //brcb
                Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(rowNum /8), {1, 8});
                // DumpTensor(tensorBetaBrcbFP32, 0,  8 * rowNum);
                PipeBarrier<PIPE_V>();
                //AscendC::printf("165\n");
                //mul
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < K) {
                    Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                        FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                // DumpTensor(tensorKFp32, 1,  K * rowNum);
                PipeBarrier<PIPE_V>();
                Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * rowNum);

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

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwddAVectorProcess<kType, betaType>::ProcessDkb() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbRowNum;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    //init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dkbInQue, 2, rowNum * sizeof(kType));
    pipe->InitBuffer(dkOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dBetaOutQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkbFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(dkFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(sharedTmpBuf, BT * sizeof(float32_t));

    auto tensorDkbFp32 = dkbFp32Buf.Get<float32_t>();
    auto tensorDkFp32 = dkFp32Buf.Get<float32_t>();
    auto tensorKfp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorSharedTmp = sharedTmpBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0;rowOffset < BT; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                //copyin
                {
                    auto tensorDkbin = dkbInQue.AllocTensor<kType>();
                    DataCopy(tensorDkbin, workSpaceTensor[kOffset], K * rowNum);
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKin, kTensor[kOffset], K * rowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);
                    auto tensorDkin = dkInQue.AllocTensor<kType>();
                    DataCopy(tensorDkin, dkTensor[kOffset], K * rowNum);
                    
                    dkbInQue.EnQue(tensorDkbin);
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    dkInQue.EnQue(tensorDkin);
                    //AscendC::printf("copyin\n");
                }
                //compute
                {
                    auto tensorDkbin = dkbInQue.DeQue<kType>();
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorDkin = dkInQue.DeQue<kType>();

                    auto tensorDkOut = dkOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    //cast fp32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetafp32, tensorBetain, rowNum);
                    }
                    Cast(tensorKfp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                    Cast(tensorDkbFp32, tensorDkbin, RoundMode::CAST_NONE, K * rowNum);
                    Cast(tensorDkFp32, tensorDkin, RoundMode::CAST_NONE, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb
                    Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    //mul
                    Mul(tensorKfp32, tensorKfp32, tensorDkbFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbFp32[perchannelResOffset], tensorDkbFp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    // DumpTensor(tensorKfp32, 1,  K * rowNum);
                    //reducesum
                    for(uint32_t row = 0;row < rowNum; row++) {
                        PipeBarrier<PIPE_V>();
                        ReduceSum(tensorKfp32, tensorKfp32[row * K], tensorSharedTmp,
                            FP32_PER_REPEAT_64, K / FP32_PER_REPEAT_64, 8);
                        SetFlag<AscendC::HardEvent::V_S>(0);
                        WaitFlag<AscendC::HardEvent::V_S>(0);
                        tensorBetafp32.SetValue(row, tensorKfp32.GetValue(0));
                    }
                    // ADD
                    PipeBarrier<PIPE_V>();
                    Add(tensorDkFp32, tensorDkFp32, tensorDkbFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, K * rowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorBetafp32, RoundMode::CAST_RINT, rowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorBetafp32, rowNum);
                    }

                    dkbInQue.FreeTensor(tensorDkbin);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    dkInQue.FreeTensor(tensorDkin);
                    dkOutQue.EnQue(tensorDkOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                }
                //copyout
                {
                    auto tensorDkOut = dkOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * rowNum);
                    DataCopy(dbetaTensor[betaOffset], tensorDbetaOut, rowNum);
                    dkOutQue.FreeTensor(tensorDkOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                }
                
            }
            
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    return;
}

#endif  // PREPARE_WY_REPR_BWD_DA_VECTOR_H

