
def init_random(seed=0):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)

from PIL import Image, ImageStat
import numpy as np

MONOCHROMATIC_MAX_VARIANCE = 0.005

def is_monochromatic_image(pil_img, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img)
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            # print("grayscale\t")
            return True
        # else:
            # print("Color\t\t\t")
        # print("( MSE=", MSE, ")")
    elif len(bands) == 1:
        # print("Black and white", bands)
        return True
    else:
        print("Don't know...", bands)
    return False
