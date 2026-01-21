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
 * \file chunk_bwd_dv_local_tiling.cpp
 * \brief
 */

#include "chunk_bwd_dv_local_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;

namespace optiling {


ge::graphStatus Tiling4ChunkBwdDvLocal(gert::TilingContext *context)
{
    std::cout<<"Tiling4ChunkBwdDvLocal"<<std::endl;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkBwdDvLocal(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDvLocal)
    .Tiling(Tiling4ChunkBwdDvLocal)
    .TilingParse<ChunkBwdDvLocalCompileInfo>(TilingPrepareForChunkBwdDvLocal); // 向框架注册入口函数

} // namespace optiling
