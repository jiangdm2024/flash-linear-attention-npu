clear
export ASCEND_RT_VISIBLE_DEVICES=2
op=chunk_bwd_dv_local

# source /data/zs/Ascend/ascend-toolkit/set_env.sh
source  /data/yzq/CANN0124/ascend-toolkit/set_env.sh
# source /data/wnc/cann/set_env.sh
# source /data/yxj/cann_0210/ascend-toolkit/set_env.sh
cd /data/clx/flash-linear-attention-npu/

# ############################## custom编译安装 ##############################
# rm -rf build
# bash build.sh --pkg --soc=ascend910b --ops=$op
# export TMPDIR=/data/clx/tmp
# ./build/cann-ops-transformer-custom_linux-aarch64.run  --install-path=/data/clx/transformer_custom

source /data/clx/transformer_custom/vendors/custom_transformer/bin/set_env.bash

############################## 执行example ##############################
# source /data/yyd/set_log.sh
# rm -rf out.log
# bash build.sh --run_example $op eager cust >& out.log

# bash build.sh --run_example $op eager cust

############################## 执行ATK ##############################
# ATK_PATH=/data/clx/ATK/chunk_bwd_dv_local

# atk node --backend PYACLNN --devices 0 node --backend CPU \
# task -c ${ATK_PATH}/all_aclnn_chunk_bwd_dv_local.json \
# -p ${ATK_PATH}/executor_chunk_bwd_dv_local.py \
# --task accuracy  -e 1

############################## 编译执行PTA ##############################
# cd /data/clx/op-plugin
# bash ci/build.sh --python=3.8 --pytorch=v2.1.0-7.1.0
# pip3 install --upgrade dist/torch_npu-2.1.0.post16-cp38-cp38-linux_aarch64.whl 
# ulimit -c unlimited

# python /data/clx/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta/test_single.py

# python /data/clx/flash-linear-attention-npu/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta/test_all.py
python /data/clx/flash-linear-attention-npu/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta/test_single.py

# python  /data/zs/ops-transformer_GDN/test.py 

############################## msprof ##############################
# cd /data/clx/ops-transformer_GDN/output
# source /data/zs/run/8.5/ascend-toolkit/set_env.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/zs/run/8.5/cann-8.5.0/aarch64-linux/simulator/Ascend910B1/lib

# msprof op simulator  --application="python3 /data/clx/flash-linear-attention-npu/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta/test_single.py"
# msprof op  --application="python3 /data/clx/flash-linear-attention-npu/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta/test_single.py"



# pkill -f "/data/clx/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dv_local/tests/pta_test.py"