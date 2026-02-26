import torch
import torch_npu
from typing import Optional
import math
import ct
import pandas as pd
import re

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor, b: int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:

    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_cu_seqlens(T: int, L: int = 32, seed: int = 42) -> torch.LongTensor:
    """
    生成一个长度为 L 的 cu_seqlens 张量：
      - 以 0 开头，以 T 结尾
      - 严格单调递增，无重复
      - 所有值在 [0, T] 范围内
      - 可复现（固定随机种子）

    Args:
        T (int): 最大值（总 token 数）
        L (int): 输出张量的长度（必须满足 2 <= L <= T + 1）
        seed (int): 随机种子，默认 42

    Returns:
        torch.LongTensor of shape [L]
    """
    if T < 1:
        raise ValueError("T must be at least 1.")
    if L < 2 or L > T + 1:
        raise ValueError(f"L must satisfy 2 <= L <= T + 1 (got L={L}, T={T}).")

    torch.manual_seed(seed)

    if L == 2:
        return torch.tensor([0, T], dtype=torch.long)

    candidates = torch.arange(1, T)

    indices = torch.randperm(T - 1)[:L - 2]
    middle_points = candidates[indices]
    middle_points = torch.sort(middle_points).values

    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        middle_points,
        torch.tensor([T], dtype=torch.long)
    ])

    return cu_seqlens.long()

def create_incremental_tensor(shape, dtype=torch.float16, start=1, step=1):
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    tensor = torch.arange(
        start,
        start + total_elements * step,
        step,
        dtype=dtype
    ).reshape(shape)
    return tensor

def bool_matrix_lower_tri_to_uint8(chunk_size):
    bool_matrix = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1)
    bool_matrix = ~bool_matrix
    uint8_matrix = bool_matrix.to(torch.uint8)
    reshaped = uint8_matrix.reshape(chunk_size, chunk_size // 8, 8)
    powers = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8)
    packed = (reshaped * powers).sum(dim=-1).to(torch.uint8)
    return packed

def get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens is not None:
        seqIdx = chunk_indices[idx][0]
        chunkIdx = chunk_indices[idx][1]
        bos = cu_seqlens[seqIdx] + chunkIdx * chunk_size
        eos = bos + chunk_size
        if eos > cu_seqlens[seqIdx + 1]:
            eos = cu_seqlens[seqIdx + 1]
    else:
        bos = idx * chunk_size
        eos = bos + chunk_size
        if eos > T:
            eos = T
    return bos, eos

def compute_dA_cpu(
    A: torch.Tensor,
    dw: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    du: torch.Tensor,
    chunk_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    BT: int,
    NT: int,
) -> torch.Tensor:
    dA = torch.zeros_like(A)
    IS_VARLEN = cu_seqlens is not None
    for idx in range(NT):
        bos, eos = get_bos_eos(idx, T, BT, cu_seqlens, chunk_indices)
        chunk_len = eos - bos
        if IS_VARLEN:
            seq_idx = chunk_indices[idx, 0].item()
            chunk_idx = chunk_indices[idx, 1].item()
            i_t = chunk_idx
            T = cu_seqlens[seq_idx + 1].item() - cu_seqlens[seq_idx].item()
        else:
            i_t = idx

        o_t = i_t * BT + torch.arange(0, BT, dtype=torch.int32)
        m_t = o_t < T
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

        for i_b in range(B):
            for i_h in range(H):
                dw_chunk = dw[i_b, i_h, bos : eos, :]
                k_chunk = k[i_b, i_h, bos : eos, :]
                beta_chunk = beta[i_b, i_h, bos : eos]
                g_chunk = g[i_b, i_h, bos : eos]

                du_chunk = du[i_b, i_h, bos : eos, :]
                v_chunk = v[i_b, i_h, bos : eos, :]

                A_chunk = A[i_b, i_h, bos : eos, : chunk_len]

                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))

                b_k_beta_g = k_chunk.to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:, None]
                if chunk_len == 1:
                    b_dA_1 = torch.sum(dw_chunk.to(torch.float32) * b_k_beta_g.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_1 = torch.matmul(dw_chunk.to(torch.float32), b_k_beta_g.T.to(torch.float32))

                b_v_beta = v_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]
                if chunk_len == 1:
                    b_dA_2 = torch.sum(du_chunk.to(torch.float32) * b_v_beta.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_2 = torch.matmul(du_chunk.to(torch.float32), b_v_beta.T.to(torch.float32))

                b_dA_3 = b_dA_1 + b_dA_2

                b_dA_4 = torch.where(m_A[:chunk_len, :chunk_len], b_dA_3.to(torch.float32), 0.0)

                if chunk_len == 1:
                    b_dA_5 = torch.sum(b_dA_4.to(torch.float32) * A_chunk.T.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_5 = torch.matmul(b_dA_4.to(torch.float32), A_chunk.T.to(torch.float32))

                if chunk_len == 1:
                    b_dA_6 = torch.sum(A_chunk.T.to(torch.float32) * b_dA_5.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_6 = torch.matmul(A_chunk.T.to(torch.float32), b_dA_5.to(torch.float32))

                b_g_sub_exp = torch.exp(g_chunk.to(torch.float32)[:, None] - g_chunk.to(torch.float32)[None, :])

                b_dA_7 = -b_dA_6.to(torch.float32) * b_g_sub_exp.to(torch.float32)

                b_dA = torch.where(m_A[:chunk_len, :chunk_len], b_dA_7.to(torch.float32), 0.0)

                dA[i_b, i_h, bos : eos, : chunk_len] = b_dA.to(A.dtype)

    return dA

def parse_shape_and_dtype(shape_str):
    match = re.match(r'\(([^)]+)\),?\s*(\w+)', shape_str)
    if match:
        shape_part = match.group(1)
        dtype_str = match.group(2)
        shape = tuple(map(int, shape_part.split(',')))
        dtype_map = {
            'FP16': torch.float16,
            'BF16': torch.bfloat16,
            'FP32': torch.float32,
            'INT64': torch.int64
        }
        dtype = dtype_map.get(dtype_str, torch.float16)
        return shape, dtype
    return None, None

def parse_cu_seqlens_shape(shape_str):
    if shape_str == '-':
        return None
    match = re.match(r'\((\d+),?\)', shape_str)
    if match:
        return int(match.group(1))
    return None

def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)

def run_test_case(case_idx, B, H, T, K, V, chunk_size, k_dtype, v_dtype, beta_dtype, A_dtype, 
                  dw_dtype, du_dtype, g_dtype, is_variable, L=None):
    print(f"\n==== Running Case {case_idx} ====")
    print(f"Params: B={B}, H={H}, T={T}, K={K}, V={V}, chunk_size={chunk_size}")
    print(f"Data types: k={k_dtype}, v={v_dtype}, beta={beta_dtype}, A={A_dtype}, dw={dw_dtype}, du={du_dtype}, g={g_dtype}")
    print(f"Type: {'Variable length' if is_variable else 'Fixed length'}")

    BT = chunk_size

    k = create_tensor((B, H, T, K), dtype=k_dtype)
    v = create_tensor((B, H, T, V), dtype=v_dtype)
    beta = create_tensor((B, H, T), dtype=beta_dtype)
    A = create_tensor((B, H, T, BT), dtype=A_dtype)
    dw = create_tensor((B, H, T, K), dtype=dw_dtype)
    du = create_tensor((B, H, T, V), dtype=du_dtype)
    g = create_tensor((B, H, T), dtype=g_dtype)

    lower_tri_matrix = bool_matrix_lower_tri_to_uint8(chunk_size)

    cu_seqlens = None
    chunk_indices = None

    if is_variable and L is not None:
        cu_seqlens = prepare_cu_seqlens(T=T, L=L)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()
    lower_tri_matrix_npu = lower_tri_matrix.npu()

    cu_seqlens_npu = None
    chunk_indices_npu = None
    if cu_seqlens is not None:
        cu_seqlens_npu = cu_seqlens.npu()
        chunk_indices_npu = chunk_indices.npu()

    dA_npu = torch_npu.npu_prepare_wy_repr_bwd_da(
        k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu,
        lower_tri_matrix=lower_tri_matrix_npu,
        cu_seqlens=cu_seqlens_npu,
        chunk_indices=chunk_indices_npu,
        chunk_size=chunk_size
    )

    # torch.save(dA_npu, f"/data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_{case_idx}_npu.pt")

    print(f"==== dA_npu.shape = {dA_npu.shape} ")

    if cu_seqlens is not None:
        NT = len(chunk_indices)
    else:
        NT = (T + BT - 1) // BT

    print("==== NT = ", NT)
    dA_cpu = compute_dA_cpu(A, dw, g, beta, k, v, du, chunk_indices, cu_seqlens, B, H, T, K, BT, NT)

    # torch.save(dA_cpu, f"/data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_{case_idx}_cpu.pt")

    try:
        ct.isclose(dA_cpu, dA_npu, diff_thd=0.1)
        print(f"==== Case {case_idx}: PASSED")
        return True
    except Exception as e:
        print(f"==== Case {case_idx}: FAILED")
        print(f"Error: {e}")
        return False

def read_cases_from_xlsx(file_path):
    df = pd.read_excel(file_path)
    cases = []
    for idx, row in df.iterrows():
        case_dict = row.to_dict()
        cases.append(case_dict)
    return cases

def main():
    torch.manual_seed(0)

    cases_file = '/data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/cases.xlsx'
    cases = read_cases_from_xlsx(cases_file)

    print(f"Total test cases: {len(cases)}")

    passed = 0
    failed = 0
    results = []

    for case_idx, case in enumerate(cases, start=1):
        try:
            k_shape, k_dtype = parse_shape_and_dtype(case['k (B,H,T,K)'])
            v_shape, v_dtype = parse_shape_and_dtype(case['v (B,H,T,V)'])
            beta_shape, beta_dtype = parse_shape_and_dtype(case['beta (B,H,T)'])
            A_shape, A_dtype = parse_shape_and_dtype(case['A (B,H,T,chunk_size)'])
            dw_shape, dw_dtype = parse_shape_and_dtype(case['dw (B,H,T,K)'])
            du_shape, du_dtype = parse_shape_and_dtype(case['du (B,H,T,V)'])
            g_shape, g_dtype = parse_shape_and_dtype(case['g (B,H,T)'])

            B, H, T, K = k_shape
            V = v_shape[3]
            chunk_size = int(case['chunk_size'])

            cu_seqlens_str = case['cu_seqlens (N+1)']
            is_variable = cu_seqlens_str != '-'
            L = parse_cu_seqlens_shape(cu_seqlens_str) if is_variable else None

            success = run_test_case(
                case_idx, B, H, T, K, V, chunk_size,
                k_dtype, v_dtype, beta_dtype, A_dtype,
                dw_dtype, du_dtype, g_dtype,
                is_variable, L
            )

            if success:
                passed += 1
            else:
                failed += 1

            results.append((case_idx, success, case.get('描述', '')))

        except Exception as e:
            print(f"==== Case {case_idx}: ERROR")
            print(f"Error: {e}")
            failed += 1
            results.append((case_idx, False, str(e)))

    print("\n" + "="*50)
    print("Summary:")
    print(f"Total: {len(cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*50)

    print("\nDetailed Results:")
    for case_idx, success, desc in results:
        status = "PASSED" if success else "FAILED"
        print(f"Case {case_idx}: {status} - {desc}")

if __name__ == "__main__":
    main()
