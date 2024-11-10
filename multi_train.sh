# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com

# accelerate launch --main_process_port 29500 --config_file ./run_config.yaml train_followyourpose.py \
#     --config="configs/pose_train.yaml" \

# accelerate launch --main_process_port 29500 --config_file ./run_config.yaml train_followyourpose.py \
#     --config="configs/temp_train.yaml" \

accelerate launch --main_process_port 29500 --config_file ./run_config.yaml train_followyourpose.py \
    --config="configs/pose_temp_train.yaml" \

# nohup bash multi_train.sh>output.log 2>&1 & disown 
