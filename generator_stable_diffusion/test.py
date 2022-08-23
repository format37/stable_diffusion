import os
import torch
from diffusers import StableDiffusionPipeline
import logging
from torch import autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Loading the model")

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=os.environ.get("HUGGINGFACE_USE_AUTH_TOKEN", False),
    # local_files_only=True,
    cache_dir='/app/data/models/'
    )
logger.info("Model loading complete")
pipe = pipe.to("cuda")

prompt = "Cyberpunk city. Pastel painting"
logger.info("Prompt:"+prompt)
with autocast("cuda"):
  image = pipe(prompt)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can do either save it such as:
image.save("data/generated/0.png")