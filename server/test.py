import os
import torch
from diffusers import StableDiffusionPipeline
import logging
from torch import autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Loading the model")

model_id = os.environ.get("MODEL_ID")
use_auth_token = os.environ.get("HUGGINGFACE_USE_AUTH_TOKEN", False)

if use_auth_token == False:
  logger.info("Auth token is not set, using local files only")
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=None,
    local_files_only=True,
    cache_dir='/app/data/models/'
    )
else:
  logger.info("Used auth token. Downloading model from huggingface.co")
  # make sure you're logged in with `huggingface-cli login`
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=os.environ.get("HUGGINGFACE_USE_AUTH_TOKEN", False),
    # local_files_only=True,
    cache_dir='/app/data/models/'
    )
logger.info("Model loading complete")
pipe = pipe.to("cuda")

prompt = "Cyberpunk santa. Pastel painting. Dramatic sky. 4k"
logger.info("Prompt:"+prompt)
with autocast("cuda"):
  image = pipe(prompt)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can do either save it such as:
image.save("data/generated/0.png")