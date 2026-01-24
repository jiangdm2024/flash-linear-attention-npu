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
 * \file chunk_gated_delta_rule_bwd_dhu_base.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
#include "kernel_operator.h"

using namespace AscendC;
namespace ChunkGDRBwdDhu {

template <typename DT>
class GDRVecBase {
public:
    __aicore__ inline GDRVecBase(){};

protected:
    __aicore__ inline void Process();
    __aicore__ inline void InitTilingData(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);


    TPipe pipe;
    TBuf<TPosition::VECCALC> calcTbuf;
    
    // inputGm
    GlobalTensor<DT> qGm;
    GlobalTensor<DT> gGm;
    GlobalTensor<DT> dvGm;
    // output gm, also used as input
    GlobalTensor<DT> dv2Gm;
    GlobalTensor<DT> dhGm;
    // inprocess workspace gm
    GlobalTensor<DT> bdvGm;
    GlobalTensor<DT> wv2Gm;
    GlobalTensor<DT> qdoGm;
    GlobalTensor<DT> gatedQGm;
    
    // calc gated q
    LocalTensor<DT> qLocal; // [BT/2,K]
    LocalTensor<float> qCastLocal;
    LocalTensor<DT> gLocal; // [BT/2,]
    LocalTensor<float> gCastLocal;
    LocalTensor<float> gExpCastLocal;
    
    // update dv2
    LocalTensor<DT> dvLocal; // [BT/2,V]
    LocalTensor<float> dvCastLocal;
    LocalTensor<DT> bdvLocal; // [BT/2,V]
    LocalTensor<float> bdvCastLocal;

    // updated dh
    LocalTensor<DT> bdhLocal; // [K/2,V]
    LocalTensor<float> bdhCastLocal;
    LocalTensor<DT> wv2Local; // [K/2,V]
    LocalTensor<float> wv2CastLocal;
    LocalTensor<DT> qdoLocal; // [K/2,V]
    LocalTensor<float> qdoCastLocal;
    
    // tiling data
    uint64_t B = 0;
    uint64_t H = 0;
    uint64_t T = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t seqNum = 0;
    uint64_t gBufSize = 0;
    uint64_t dvBufSize = 0;
    uint64_t qBufSize = 0;
    uint64_t dhBufSize = 0;
    uint64_t totalTbufByte = 0;
    uint64_t isVarLen = 0;
    uint64_t isScale = 0;
    uint32_t usedCoreNum = 0;
    float  scale = 0;
};

template <typename DT>
__aicore__ inline void GDRVecBase<DT>::InitTilingData(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    this->B = tilingData.B;
    printf("B is %llu \n",this->B);
}



}  // namespace ChunkGDRBwdDhu
#endif  // CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
