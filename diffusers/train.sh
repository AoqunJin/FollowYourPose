export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export data_dir="/home/sora/workspace/diffusers/hf_data/train"

accelerate launch --mixed_precision="bf16" --num_processes=1 --gpu_ids '0' train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$data_dir \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --mixed_precision="bf16" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="metaworld"