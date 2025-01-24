from PIL import Image
from torchvision import transforms
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import re

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512))
    img = transforms.ToTensor()(img).unsqueeze_(0)
    img = img * 2.0 - 1.0
    img = img.half()
    return img

def clear_dir(path):
    for f in path.iterdir():
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

def save_image(img, path, tag):
    img = img.clone()
    img = img - img.min()
    img = img / img.max()
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))


    threshold = 0.75

    # print(att_maps.max())
    # print(att_maps.min())
    img[img < threshold] *= 0.9
    # img[img >= threshold] *= 1.0
    # Clip the values
    # np.clip(img, 0, 255, out=img)

    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)
    if img[0].shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    img = Image.fromarray(img)
    img = img.resize((512, 512), Image.BICUBIC)
    img.save(f"{path}/{tag}.png")
    return img

def load_mask(path):
    mask = Image.open(path)
    mask = mask.convert("L")
    mask = mask.resize((64, 64))
    mask = np.array(mask)
    mask = mask / 255.0
    mask = torch.tensor(mask).half()
    return mask

def save_att(att_map, token_id):
    vis_map = att_map[:, token_id, :, :]
    vis_map = vis_map.mean(dim=0).unsqueeze(0).unsqueeze(0)
    vis_map = F.interpolate(vis_map, size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)



    return vis_map


def custom_threshold(image, threshold, alpha, beta):
    # Ensure alpha and beta are set correctly
    assert 0 <= alpha <= 1 and 1 <= beta, "Alpha should be in [0, 1] and Beta should be >= 1"

    # Create a copy of the image to avoid modifying the original
    transformed_image = np.copy(image)

    # Applying custom transformation
    transformed_image[transformed_image < threshold] *= alpha
    transformed_image[transformed_image >= threshold] *= beta

    # Clipping values to ensure they remain within [0, 255]
    np.clip(transformed_image, 0, 255, out=transformed_image)

    # Convert back to uint8
    transformed_image = transformed_image.astype(np.uint8)

    return transformed_image

def strip_style(text):
    # This regex pattern looks for any phrase that starts with 'with ' and continues until the end of the string.
    pattern = r' with .*$'
    # The re.sub function replaces the matched pattern with an empty string, effectively removing it.
    return re.sub(pattern, '', text)
