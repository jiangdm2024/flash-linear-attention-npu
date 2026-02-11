path=/root/data_nvme0n1/huangjunzhe/GDN/target/result/cpu_for_test
source /root/data_nvme0n1/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
# conda activate gdn_py39
compi=$1
compi_y="all"

if [ "$compi" = "$compi_y" ]; then
    python3 /root/data_nvme0n1/huangjunzhe/GDN/target/test2.py  #标杆生成pt
    python3 /root/data_nvme0n1/huangjunzhe/GDN/target/result/pre_handle.py ${path} # pt -> bin
    cd /root/data_nvme0n1/huangjunzhe/GDN/code/old/
    bash run.sh compile  ##重新编译并运行/root/data_nvme0n1/huangjunzhe/GDN/target/test_aclnn_gdn.cpp
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /root/data_nvme0n1/huangjunzhe/GDN/target/result/to_pt.py ${path} # bin -> pt
echo "ct single ${path}/gen/dg_npu.pt ${path}/dg_cpu.pt --calc_count 1000000 --dtype float16"
ct single ${path}/gen/dw_npu.pt ${path}/gen/dw_cpu_ht.pt --calc_count 1000000 --dtype float16
ct single ${path}/gen/dg_npu.pt ${path}/gen/dg_cpu_ht.pt --calc_count 1000000 --dtype float16
ct single ${path}/gen/dq_npu.pt ${path}/gen/dq_cpu_ht.pt --calc_count 1000000 --dtype float16
ct single ${path}/gen/dk_npu.pt ${path}/gen/dk_cpu_ht.pt --calc_count 1000000 --dtype float16