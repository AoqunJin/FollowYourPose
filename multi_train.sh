export CUDA_VISIBLE_DEVICES="2,3"

TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 29501 --config_file ./run_config.yaml train_followyourpose.py \
    --config="configs/pose_train.yaml" \

# TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 29501 --config_file ./run_config.yaml train_followyourpose.py \
#     --config="configs/temp_train.yaml" \

# nohup  bash multi_train.sh>output.log 2>&1 & disown 
