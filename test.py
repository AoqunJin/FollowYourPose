import pandas as pd
import torch

from followyourpose.models.unet import UNet3DConditionModel, UNet2DConditionModel


unet = UNet2DConditionModel.from_pretrained_2d("/home/ao/workspace/FollowYourPose/diffusers/metaworld-1002-1800", subfolder="unet")

unet.load_state_dict(torch.load("/home/ao/workspace/FollowYourPose/output-1004-0800/checkpoint-30000/pytorch_model.bin"))

torch.save(unet.skeleton_adapter.state_dict(), "/home/ao/workspace/FollowYourPose/output-1004-0800/checkpoint-30000/skeleton_adapter.bin")

