import os
import torch
from diffusers import StableDiffusionPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
id = "model_id"
model_path = f"/path/to/model/{id}"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")


test_texts = [
    "Vertical button",    
    "Vertical button, Reddish-brown wall, (Wall)", 
    "Horizontal button", 
    "Horizontal button, Reddish-brown wall, (Wall)", 
    "Red Cylinder, Red target point", 
    "Red Cylinder, Red target point, Reddish-brown wall, (Wall)", 
    "Red Cylinder, Green target point", 
    "Red Cylinder, Green target point, Reddish-brown wall, (Wall)",
    "Red Cylinder, Blue target point",
    "Red Cylinder, Blue target point, Reddish-brown wall, (Wall)",
    "Brown ferrule and green handle, Red Cylinder", 
    "Black metal safety door",
    "Black metal safety door, Door close, (Close)",  
    "Green drawer",
    "Grey faucet with red handle", 
    "White soccer net, Black plate", 
    "White soccer net side, Black plate, (Side)",
    "Nice glass window",
]

test_texts = ["Vertical button"] * 15
test_texts = ["Horizontal button"] * 15

print("Total", len(test_texts))

for i, p in enumerate(test_texts):
    image = pipe(prompt=p, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"/path/to/model/{id}/inference/{i}.png")
    