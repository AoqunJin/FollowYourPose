import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import csv
import cv2

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import PartialState

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from followyourpose.models.unet import UNet3DConditionModel
from followyourpose.data.hdvila import HDVilaDataset
from followyourpose.pipelines.pipeline_followyourpose import FollowYourPosePipeline
from followyourpose.util import save_videos_grid, ddim_inversion
from einops import rearrange

import sys
sys.path.append('FollowYourPose')

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# Function to read CSV and return prompts and skeleton paths
def read_csv(csv_file):
    prompts = []
    skeleton_paths = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skeleton_paths.append(row['pose_video'])
            prompts.append(row['text'])
    return prompts, skeleton_paths

def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    
    return frame_count

# Function to process videos with the pipeline
def process_videos(csv_file, validation_pipeline, validation_data, generator, ddim_inv_latent, output_dir, global_step, seed):
    # Read data from CSV
    prompts, skeleton_paths = read_csv(csv_file)
    # samples = []
    
    for idx, prompt in enumerate(prompts):
        
        now = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        skeleton_path = skeleton_paths[idx]
        sample_frame_rate = 4
        validation_data["video_length"] = get_frame_count(skeleton_path) // sample_frame_rate
        
        while validation_data["video_length"] > 24:
            sample_frame_rate += 1
            validation_data["video_length"] = get_frame_count(skeleton_path) // sample_frame_rate
            
        # Process video with validation pipeline
        sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent,
                                     skeleton_path=skeleton_path, frame_skeleton_stride=sample_frame_rate, **validation_data).videos
        
        # Save individual sample video as GIF
        p = skeleton_path.split("/")[-1]
        sample_output_dir = f"{output_dir}/inference/{p}.gif"
        os.makedirs(os.path.dirname(sample_output_dir), exist_ok=True)
        save_videos_grid(sample, sample_output_dir)
        # samples.append(sample)
    
    # Concatenate all samples and save as a single GIF
    # samples = torch.concat(samples)
    # save_path = f"{output_dir}/inference/sample-{global_step}-{str(seed)}-{now}.gif"
    # save_videos_grid(samples, save_path)
    # print(f"Saved samples to {save_path}")
    
def main(
    pretrained_model_path: str,
    output_dir: str,
    num_inv_steps: int,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None
):
    *_, config = inspect.getargvalues(inspect.currentframe())


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # output_dir = os.path.join(output_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    
    ddim_inv_scheduler.set_timesteps(num_inv_steps)
    
    if resume_from_checkpoint:
        unet.load_state_dict(torch.load(resume_from_checkpoint))

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(weight_dtype)
    vae.to(weight_dtype)
    unet.to(weight_dtype)

    # Get the validation pipeline
    validation_pipeline = FollowYourPosePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    
    distributed_state = PartialState()
    validation_pipeline.to(distributed_state.device)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.


    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    load_path = None
    # if resume_from_checkpoint:
    #     if resume_from_checkpoint != "latest":
    #         load_path = resume_from_checkpoint
    #         output_dir = os.path.abspath(os.path.join(resume_from_checkpoint, ".."))
    #     accelerator.print(f"load from checkpoint {load_path}")
    #     accelerator.load_state(load_path)

    #     global_step = int(load_path.split("-")[-1])

    # Example usage
    csv_file = '/home/ao/workspace/meta_caption_pr_B_f_2.csv'  # Update with actual path
    output_dir = '/home/ao/workspace/tmp/output-1006-1334/checkpoint-20000'       # Update with actual path
    global_step = 1000                       # Replace with actual global step
    seed = 42                                # Replace with your seed

    validation_data = {
        "video_length": 24,
        "width": 512,
        "height": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    ddim_inv_latent = None

    process_videos(csv_file, validation_pipeline, validation_data, generator, ddim_inv_latent, output_dir, global_step, seed)


if __name__ == "__main__":
    main(
        pretrained_model_path="/home/ao/workspace/tmp/metaworld-1002-1800",
        resume_from_checkpoint="/home/ao/workspace/tmp/output-1006-1334/checkpoint-20000/pytorch_model.bin",
        output_dir="output",
        num_inv_steps=50
    )
