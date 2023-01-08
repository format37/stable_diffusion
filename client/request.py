import requests
from datetime import datetime as dt
import time
# import json
import matplotlib.image as mpimg
from io import BytesIO
import matplotlib.pyplot as plt
import random
import logging
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    server_address = 'http://localhost:10000'
    logger.info('Checking for server to be ready..')
    r = ''
    while r == '':
        try:
            r = requests.get(server_address+'/test').text
        except Exception as e:
            time.sleep(1)
    logger.info(str(r))
    logger.info('Server is ready')

    seed = random.randint(0, 1000000)
    logger.info('Seed: {}'.format(seed))
    # width & height available options:
    # 512, 576, 640, 704, 768, 832, 896, 960, 1024
    request = {
                'prompt': "Happy cat in a cyberpunk city, photorealistic, hdr, 4k, wallpaper, epic, cinematic, ray-traced, 8k, unigine render",
                'width': 768,
                'height': 512,
                'num_inference_steps': 30,
                'guidance_scale': 7,
                'seed': seed
            }

    logger.info('Requesting..')
    start_time = dt.now()
    # Do request with the long wait period
    response = requests.post(server_address+'/request', json=request, timeout=60)
    if response.status_code==200:
        logger.info('200 - Ok')
        
        # The responce will contains an image, which will be sent like:
        # image_data = BytesIO()
        # image.save(image_data, format='png')
        # image_data.seek(0)
        # return Response(image_data, mimetype='image/png')

        # Therefore we need to receive it and save it to dis
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.save('image.png')
        logger.info('Image saved')
        
    else:
        logger.error(response.status_code)
        logger.error(response.text)

    end_time = dt.now()
    logger.info('Request took: {}'.format(end_time-start_time))
    
    logger.info('Done')
    
if __name__ == "__main__":
    main()
