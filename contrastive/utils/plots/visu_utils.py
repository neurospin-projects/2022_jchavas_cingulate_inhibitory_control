#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)


def buffer_to_image(buffer):
    """Transforms IO buffer into tensor representing PNG image"""

    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close('all')
    image = PIL.Image.open(buffer)
    image = ToTensor()(image).unsqueeze(0)[0]
    return image

def png_file_to_image(png_file):
    """Transforms PNG file into tensor representing PNG image"""

    image = PIL.Image.open(png_file)
    image = ToTensor()(image).unsqueeze(0)[0]
    return image


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
