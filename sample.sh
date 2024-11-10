export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export HF_ENDPOINT=https://hf-mirror.com

python inference.py

# nohup bash sample.sh>output.log 2>&1 & disown 
