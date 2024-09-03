#!/bin/bash

umask 0002

cd /proj/data-eng/blog/torchtitan
startTime=$(date +%s) 

#== gather allocated resource info:
MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=28444 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
JOB_ID=${LSB_JOBID}

hostname=`hostname`
runtime=$(date "+%Y-%m-%d-%H-%M")
if [ -z $GPUS_PER_NODE ]; then GPUS_PER_NODE=8; fi

consolefile=/proj/data-eng/blog/torchtitan/outputs/train_${NODE_RANK}.log
export NCCL_DEBUG_FILE="/proj/data-eng/blog/torchtitan/outputs/NCCL_DEBUG_FILE.${runtime}.%h"

#=========== TRACKING TRAINING PARAMS + ENV:
echo " " | tee -a $consolefile # ovewrite prev.file if existed
echo "------------------------  Running Env: ------------------------ " | tee -a $consolefile
echo "JOB_ID: ${JOB_ID} " | tee -a $consolefile
echo "RUNTIME: ${runtime} " | tee -a $consolefile
echo "CONSOLE LOG: $consolefile  " | tee -a $consolefile
echo "HOSTNAME: ${hostname} " | tee -a $consolefile
echo "MASTER_ADDR: ${MASTER_ADDR} " | tee -a $consolefile
echo "MASTER_PORT: ${MASTER_PORT} " | tee -a $consolefile

# GPUs:
echo "NNODES: ${NNODES} " | tee -a $consolefile
echo "GPUS_PER_NODE: ${GPUS_PER_NODE} " | tee -a $consolefile
echo "NODE_RANK: ${NODE_RANK} " | tee -a $consolefile
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE  " | tee -a $consolefile

echo "-------------------------------------------------------------------" | tee -a $consolefile

# activate conda env:
CONDA_INIT_PATH="/opt/share/miniconda/etc/profile.d/conda.sh"
source ${CONDA_INIT_PATH}
CONDA_DIR=/opt/share/miniconda/
source $CONDA_DIR/etc/profile.d/conda.sh
echo "conda activate blog-env" | tee -a $consolefile
conda activate blog-env
let rc=$?
if [ $rc -ne 0 ]; then
    echo "Conda Activate ${CONDA_ENV_PATH} failed .. Exiting " | tee -a $consolefile
    exit $rc
fi

verbose=0
# Add more log for conda, torch to log file in verbose mode:
if [[ ${verbose} -eq 1 ]]; then
   echo -e "\n---------------- CONDA INFO: " >> $consolefile
   conda info >>$consolefile
   echo -e "\n---------------- CONDA LIST: " >> $consolefile
   conda list >>$consolefile
   echo -e "\n---------------- nvidia-smi topo: " >> $consolefile
   nvidia-smi topo -m >>$consolefile
   echo -e "\n---------------- torch.utils.collect_env: " >> $consolefile
   python -m torch.utils.collect_env >>$consolefile
   #For Debug and to prevent accidental sharing of secrets to log file, only care about NCCL, CUDA, LSF or Podman vars:
   echo -e "\n---------------- ENV VAR HPCX|LIBRARY|LD_|TOKENIZER|LSF_|LSB_|XDG_|CUD|NCCL|OMP: " >> $consolefile
   printenv | grep -E "HPCX|LIBRARY|LD_|TOKENIZER|LSF_|LSB_|XDG_|CUD|NCCL|OMP"  | sort >>$consolefile
fi

DISTRIBUTED_ARGS="\
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_3b.toml"}
echo " " | tee -a $consolefile
echo "------------------------  DISTRIBUTED_ARGS: ------------------ " | tee -a $consolefile
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS" | tee -a $consolefile
echo "-------------------------------------------------------------------" | tee -a $consolefile
echo " " | tee -a $consolefile
echo -e "\n\n---------------- START TRAINING ..." >> $consolefile
echo torchrun $DISTRIBUTED_ARGS train.py --job.config_file ${CONFIG_FILE} --float8.enable_float8_linear --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp >> $consolefile

#=== Start training
torchrun $DISTRIBUTED_ARGS train.py --job.config_file ${CONFIG_FILE} --float8.enable_float8_linear --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp >>$consolefile 2>&1
rc=$?
endTime=$(date +%s)
elapsed=$(($endTime-$startTime))
echo "== TRAINING \`$EXPERIMENT_ID\` IS DONE IN: ${elapsed}(s) rc=$rc (log file: $consolefile)" | tee -a $consolefile
