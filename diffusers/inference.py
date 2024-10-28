import os
import torch
from diffusers import StableDiffusionPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
id = "metaworld-1009-0214"
model_path = f"/home/ao/workspace/FollowYourPose/diffusers/{id}"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")


test_texts = [
    "Brown ferrule and green handle, Red Cylinder",
    "Brown ferrule and green handle, Red Cylinder, Ring outside the column, (Outside)",
    "Brown ferrule and green handle, Red Cylinder, Rings over columns, (Over columns)",

    "Black metal safety door",
    "Black metal safety door, Door open, (Open)",
    "Black metal safety door, Door close, (Close)",

    "Black metal safety door, Door close, (Close)",
    "Black metal safety door, Door close, Door lock, (Close), (Lock)",
    "Black metal safety door, Door close, Door unlock, (Close), (Unlock)",

    "Green drawer",
    "Green drawer, Drawer open, (Open)",
    "Green drawer, Drawer close, (Close)",

    "Grey faucet with red handle",
    "Grey faucet with red handle, Faucet open, (Open)",
    "Grey faucet with red handle, Faucet close, (Close)",

    "White soccer net, Black plate",
    "White soccer net, Black plate, Plate is in the net, (In the net)",
    "White soccer net, Black plate, Plate outside in the net, (Outside the net)",

    "White soccer net side, Black plate, (Side)",
    "White soccer net side, Black plate, Plate is in the net, (Side), (In the net)",
    "White soccer net side, Black plate, Plate outside in the net, (Side), (Outside the net)",

    "Nice glass window",
    "Nice glass window, Window open, (Open)",
    "Nice glass window, Window close, (Close)",
    
    "Red Cylinder, Red target point",
    "Red Cylinder, Green target point",
]

# test_texts = ["Brown ferrule and green handle, Red Cylinder"] * 20

# test_texts = ["Black metal safety door",] * 5
print("Total", len(test_texts))

for i, p in enumerate(test_texts):
    image = pipe(prompt=p, num_inference_steps=50, guidance_scale=6.5).images[0]
    image.save(f"{id}/inference-s/{i}.png")
    