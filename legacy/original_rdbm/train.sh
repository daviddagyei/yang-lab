export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

start_time=$(date +%s)
echo "start_time: ${start_time}"

nohup torchrun --nproc_per_node=8 -m legacy.original_rdbm.train \
  > ./train.log 2>&1 &
 
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))
 
echo "------------------ Final result ------------------"
