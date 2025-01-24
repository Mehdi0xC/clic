from PIL import Image
from lang_sam import LangSAM
import numpy as np  # Add this import
import sys
from utils import config_utils
import cv2
from scipy.ndimage import binary_fill_holes
from transformers import logging as hf_logging

import warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

def main(config):
    model = LangSAM()
    image_pil = Image.open("input/input.png").convert("RGB")
    text_prompt = config.prompt
    masks, _, _, _ = model.predict(image_pil, text_prompt)

    # Save the masks
    for i, mask in enumerate(masks[:1]):
        # Convert to NumPy array
        mask = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)
        mask_np = mask.astype("uint8") * 255

        if config.fill_seg_holes:
            mask_np = binary_fill_holes(mask_np // 255).astype('uint8') * 255

        if config.dilate_seg_mask:
            kernel = np.ones((config.dilation_factor,config.dilation_factor),np.uint8)
            mask_np = cv2.dilate(mask_np,kernel,iterations = 1)

        if config.erode_seg_mask:
                    kernel = np.ones((5,5),np.uint8)
                    mask_np = cv2.erode(mask_np,kernel,iterations = 1)

        if config.soft_edges:
            mask_np = cv2.GaussianBlur(mask_np,(25,25),0)

        mask_pil = Image.fromarray(mask_np, mode="L")
        mask_pil.save(f"input/mask.png")

if __name__ == "__main__":
    config = config_utils.load_config(sys.argv)
    main(config)