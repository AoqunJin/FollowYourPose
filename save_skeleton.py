import torch

from followyourpose.models.unet import UNet2DConditionModel


unet = UNet2DConditionModel.from_pretrained_2d("/path/to/sd_model", subfolder="unet")

unet.load_state_dict(torch.load("/path/to/pytorch_model.bin"))

torch.save(unet.skeleton_adapter.state_dict(), "/path/to/skeleton_adapter.bin")
