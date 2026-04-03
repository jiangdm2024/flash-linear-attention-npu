/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o.cpp
 * \brief
 */

// #include "chunk_fwd_o.h"
#include "catlass/gemm/kernel/gdn_fwd_o_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

extern "C" __global__ __aicore__ void chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                                         GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_offsets,
                                                         GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    
    __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);
    using workspaceType = float;
    // dtype: 0 - fp16, 1 - bf16, 2 - fp32
    if (gdnFwdOTilingData->dataType == 1) {
        if (gdnFwdOTilingData->gDataType == 2) {
            using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<bfloat16_t, float, workspaceType>;
            GDNFwdOKernel gdnFwdO;
            gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            gdnFwdO.Process();
        } else {
            using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<bfloat16_t, bfloat16_t, workspaceType>;
            GDNFwdOKernel gdnFwdO;
            gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            gdnFwdO.Process();
        }
    } else {
        if (gdnFwdOTilingData->gDataType == 2) {
            using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<half, float,workspaceType>;
            GDNFwdOKernel gdnFwdO;
            gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            gdnFwdO.Process();
        } else {
            using GDNFwdOKernel = Catlass::Gemm::Kernel::GDNFwdOKernel<half, half, workspaceType>;
            GDNFwdOKernel gdnFwdO;
            gdnFwdO.Init(q, k, v, h, g, cu_seqlens, chunk_offsets, o, tiling, user);
            gdnFwdO.Process();
        }
    }
}
