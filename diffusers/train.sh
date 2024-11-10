# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export data_dir="/path/to/dataset"

accelerate launch --main_process_port 29501 --config_file ./run_config.yaml train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$data_dir \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/output"

# nohup bash train.sh>output.log 2>&1 & disown 
