import torch
import torch_npu
from typing import Optional, List
import math
import hashlib
import os
from dataclasses import dataclass
from ct import single
from golden import chunk_bwd_dv_local_fix, chunk_bwd_dv_local_variable, prepare_chunk_indices
from utils import generate_cu_seqlens, create_tensor, bool_matrix_to_uint8, compare_tensors_by_ratio

golden_out_dir="/data/clx/golden_out_no_mask_input"
@dataclass
class TestCase:
    B: int
    H: int
    T: int
    K: int
    V: int
    cu_seqlens_len: Optional[int]
    chunk_num: Optional[int]
    scale: float
    chunk_size: int
    q_dtype: str  # q, k, d_o的数据类型
    g_dtype: str  # g的数据类型


def parse_case_params(file_path: str) -> List[TestCase]:
    """从 case_params.txt 解析测试用例"""
    cases = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            params = {}
            dtype_values = []
            for pair in line.split(','):
                if '=' in pair:
                    key, value = pair.split('=')
                    if value == 'None':
                        params[key] = None
                    elif key in ['B', 'H', 'T', 'K', 'V', 'cu_seqlens_len', 'chunk_num', 'chunk_size']:
                        params[key] = int(value)
                    else:
                        params[key] = float(value)
                else:
                    # 处理数据类型字段 (FP16, BF16, FP32等)
                    dtype_values.append(pair)
            # 行尾的两个数据类型分别是q_dtype和g_dtype
            if len(dtype_values) >= 2:
                params['q_dtype'] = dtype_values[-2]
                params['g_dtype'] = dtype_values[-1]
            cases.append(TestCase(**params))
    return cases


def save_dv_golden(dv_golden: torch.Tensor, case: TestCase):
    """保存 dv_golden 到 golden_out 目录，使用参数值命名文件"""
    os.makedirs(golden_out_dir, exist_ok=True)
    filename = f"dv_golden_B{case.B}_H{case.H}_T{case.T}_K{case.K}_V{case.V}_chunk{case.chunk_size}_scale{case.scale}_{case.q_dtype}_{case.g_dtype}.pt"
    save_path = os.path.join(golden_out_dir, filename)
    torch.save(dv_golden, save_path)
    print(f"==== Saved dv_golden to: {save_path}")


def load_dv_golden(case: TestCase) -> torch.Tensor:
    """从 golden_out 目录加载 dv_golden，如果文件不存在则返回 None"""
    filename = f"dv_golden_B{case.B}_H{case.H}_T{case.T}_K{case.K}_V{case.V}_chunk{case.chunk_size}_scale{case.scale}_{case.q_dtype}_{case.g_dtype}.pt"
    load_path = os.path.join(golden_out_dir, filename)
    
    if not os.path.exists(load_path):
        print(f"==== Warning: dv_golden file not found: {load_path}")
        return None
    
    dv_golden = torch.load(load_path)
    print(f"==== Loaded dv_golden from: {load_path}")
    return dv_golden


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """将字符串数据类型转换为torch.dtype"""
    dtype_map = {
        'FP16': torch.float16,
        'BF16': torch.bfloat16,
        'FP32': torch.float32,
    }
    return dtype_map.get(dtype_str.upper(), torch.float16)


def test_variable(case: TestCase):
    B, H, T, K, V = case.B, case.H, case.T, case.K, case.V
    chunk_size = case.chunk_size
    scale = case.scale
    cu_seqlens_len = case.cu_seqlens_len
    q_dtype = get_torch_dtype(case.q_dtype)
    g_dtype = get_torch_dtype(case.g_dtype)

    q = create_tensor((B, H, T, K), dtype=q_dtype)
    k = create_tensor((B, H, T, K), dtype=q_dtype)
    d_o = create_tensor((B, H, T, V), dtype=q_dtype)
    g = create_tensor((B, H, T), dtype=g_dtype)

    print(f"==== q.shape = {q.shape}, dtype = {q_dtype} ")
    print(f"==== k.shape = {k.shape}, dtype = {q_dtype} ")
    print(f"==== d_o.shape = {d_o.shape}, dtype = {q_dtype} ")
    print(f"==== g.shape = {g.shape}, dtype = {g_dtype} ")
    
    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    print(f"==== cu_seqlens = {cu_seqlens}")
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dv_golden = load_dv_golden(case)
    if dv_golden is None:
        dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)
        save_dv_golden(dv_golden, case)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, g_gamma=None, A=None, cu_seqlens=cu_seqlens_list, chunk_indices=chunk_indices_list, scale=scale, chunk_size=chunk_size)

    single(dv.cpu(),dv_golden)

def test_fix(case: TestCase):
    B, H, T, K, V = case.B, case.H, case.T, case.K, case.V
    chunk_size = case.chunk_size
    scale = case.scale
    q_dtype = get_torch_dtype(case.q_dtype)
    g_dtype = get_torch_dtype(case.g_dtype)

    q = create_tensor((B, H, T, K), dtype=q_dtype)
    k = create_tensor((B, H, T, K), dtype=q_dtype)
    d_o = create_tensor((B, H, T, V), dtype=q_dtype)
    g = create_tensor((B, H, T), dtype=g_dtype)

    print(f"==== q.shape = {q.shape}, dtype = {q_dtype} ")
    print(f"==== k.shape = {k.shape}, dtype = {q_dtype} ")
    print(f"==== d_o.shape = {d_o.shape}, dtype = {q_dtype} ")
    print(f"==== g.shape = {g.shape}, dtype = {g_dtype} ")
    cu_seqlens = None

    dv_golden = load_dv_golden(case)
    if dv_golden is None:
        dv_golden = chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)
        save_dv_golden(dv_golden, case)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, g_gamma=None, A=None,cu_seqlens=None, chunk_indices = None, scale=scale, chunk_size =chunk_size)
    single(dv.cpu(),dv_golden)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 获取 case_params.txt 的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    case_params_path = os.path.join(current_dir, "case_params.txt")

    # 解析所有测试用例
    all_cases = parse_case_params(case_params_path)

    # 遍历所有用例，根据 B 的值选择不同的测试函数
    count = 0
    for case in all_cases:
        print(f"\n{'='*60}")
        print(f"Running with case: B={case.B}, H={case.H}, T={case.T}, K={case.K}, V={case.V}")
        print(f"chunk_size={case.chunk_size}, scale={case.scale}, cu_seqlens_len={case.cu_seqlens_len}")
        print(f"q_dtype={case.q_dtype}, g_dtype={case.g_dtype}")
        print(f"{'='*60}")
        count += 1
        # if count==3:
        #     # break
        #     if case.B == 1:
        #         test_variable(case)
        #     else:
        #         test_fix(case)
        #     break
        if case.B == 1:
                test_variable(case)
        # else:
        #     test_fix(case)



    
