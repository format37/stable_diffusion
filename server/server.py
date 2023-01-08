import os
import torch
from diffusers import StableDiffusionPipeline
import logging
from torch import autocast
from flask import Flask, request, jsonify, Response
from flask import send_file
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

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

@app.route("/test")
def call_test():
  return "get ok"

@app.route("/request", methods=["POST"])
def call_request():
  try:
    r = request.get_json()
    logger.info("call_request prompt: {}".format(r))
    prompt = r["prompt"]
    height = int(r["height"])
    width = int(r["width"])
    num_inference_steps = int(r["num_inference_steps"])
    guidance_scale = r["guidance_scale"]
    guidance_scale = None if guidance_scale is None else float(guidance_scale)
    seed = int(r["seed"])
    seed = None if seed is None else int(str(seed))
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None  

    logger.info("Prompt:"+prompt)
    logger.info("Height:"+str(height))
    logger.info("Width:"+str(width))
    logger.info("Num inference steps:"+str(num_inference_steps))
    logger.info("Guidance scale:"+str(guidance_scale))
    logger.info("Seed:"+str(seed))

    with autocast("cuda"):
      image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale
        )["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

    image_data = BytesIO()
    image.save(image_data, format='png')
    image_data.seek(0)

    return Response(image_data, mimetype='image/png')
  except Exception as e:
    logger.error(e)
    # Return with code non 200 to indicate error
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=int(os.environ.get("PORT", 10000)))
