// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <iostream>  // 添加iostream头文件用于打印

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_prepare_wy_repr_bwd_da(
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &beta,
    const at::Tensor &A,
    const at::Tensor &dw,
    const at::Tensor &du,
    const at::Tensor &g,
    const at::Tensor &lower_tri_matrix,
    const c10::optional<at::Tensor> &cu_seqlens,
    const c10::optional<at::Tensor> &chunk_indices,
    int64_t chunk_size
)
{
    // 打印函数开始执行
    std::cout << "=== 开始执行 npu_prepare_wy_repr_bwd_da ===" << std::endl;
    
    // 打印输入张量的形状信息
    std::cout << "输入张量形状:" << std::endl;
    std::cout << "k.shape: [";
    for (int i = 0; i < k.dim(); ++i) {
        std::cout << k.size(i);
        if (i < k.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "v.shape: [";
    for (int i = 0; i < v.dim(); ++i) {
        std::cout << v.size(i);
        if (i < v.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "beta.shape: [";
    for (int i = 0; i < beta.dim(); ++i) {
        std::cout << beta.size(i);
        if (i < beta.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "A.shape: [";
    for (int i = 0; i < A.dim(); ++i) {
        std::cout << A.size(i);
        if (i < A.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "dw.shape: [";
    for (int i = 0; i < dw.dim(); ++i) {
        std::cout << dw.size(i);
        if (i < dw.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "du.shape: [";
    for (int i = 0; i < du.dim(); ++i) {
        std::cout << du.size(i);
        if (i < du.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "g.shape: [";
    for (int i = 0; i < g.dim(); ++i) {
        std::cout << g.size(i);
        if (i < g.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "lower_tri_matrix.shape: [";
    for (int i = 0; i < lower_tri_matrix.dim(); ++i) {
        std::cout << lower_tri_matrix.size(i);
        if (i < lower_tri_matrix.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 打印可选张量信息
    if (cu_seqlens.has_value()) {
        std::cout << "cu_seqlens.shape: [";
        for (int i = 0; i < cu_seqlens.value().dim(); ++i) {
            std::cout << cu_seqlens.value().size(i);
            if (i < cu_seqlens.value().dim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "cu_seqlens: None" << std::endl;
    }
    
    if (chunk_indices.has_value()) {
        std::cout << "chunk_indices.shape: [";
        for (int i = 0; i < chunk_indices.value().dim(); ++i) {
            std::cout << chunk_indices.value().size(i);
            if (i < chunk_indices.value().dim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "chunk_indices: None" << std::endl;
    }
    
    std::cout << "chunk_size: " << chunk_size << std::endl;
    
    // 打印准备创建输出张量
    std::cout << "准备创建输出张量 dA..." << std::endl;
    at::Tensor dA = npu_preparation::apply_tensor_without_format(A.sizes(), A.options().dtype());
    std::cout << "dA.shape: [";
    for (int i = 0; i < dA.dim(); ++i) {
        std::cout << dA.size(i);
        if (i < dA.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    const at::Tensor &cu_seqlens_ = c10::value_or_else(cu_seqlens, [] { return at::Tensor(); });
    const at::Tensor &chunk_indices_ = c10::value_or_else(chunk_indices, [] { return at::Tensor(); });
    
    // 打印开始执行NPU命令
    std::cout << "开始执行 EXEC_NPU_CMD(aclnnPrepareWyReprBwdDa)..." << std::endl;
    EXEC_NPU_CMD(aclnnPrepareWyReprBwdDa, k, v, beta, A, dw, du, g, lower_tri_matrix, cu_seqlens, chunk_indices, chunk_size, dA);
    
    // 打印执行完成
    std::cout << "EXEC_NPU_CMD 执行完成" << std::endl;
    std::cout << "=== npu_prepare_wy_repr_bwd_da 执行结束 ===" << std::endl;
    
    return dA;
}

}  // namespace op_api
