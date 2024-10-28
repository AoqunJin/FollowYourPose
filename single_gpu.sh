# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
    --num_processes=1 --gpu_ids '0' \
    train_followyourpose.py \
    --config="configs/pose_train.yaml" \

# Use to train
# nohup  bash single_gpu.sh>output.log 2>&1 & disown 
