#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import dico_toolbox as dtx
import anatomist.headless as anatomist
from soma import aims
import io
import logging

import matplotlib.pyplot as plt
from numpy import int16

from .visu_utils import buffer_to_image

logger = logging.getLogger(__name__)


a = None
win = None


class Visu_Anatomist:

    def __init__(self, ):
        global a
        global win
        a = anatomist.Anatomist()
        win = a.createWindow('3D')
        win.setHasCursor(0)

    def plot_bucket(self, img, buffer):
        """Plots as 3D buckets the first 3D image of the batch

        Args:
            img: batch of images of size [size_batch, 1, size_X, size_Y, size_Z]
            buffer (boolean): True -> returns PNG image buffer
                            False -> plots the figure
        """
        global a
        global win
        arr = img[0, 0, :, :, :]
        vol = aims.Volume(arr.numpy().astype(int16))
        bucket_map = dtx.convert.volume_to_bucketMap_aims(vol)
        bucket_a = a.toAObject(bucket_map)
        bucket_a.addInWindows(win)
        view_quaternion = [0.4, 0.4, 0.5, 0.5]
        win.camera(view_quaternion=view_quaternion)
        win.imshow(show=False)

        if buffer:
            win.removeObjects(bucket_a)
            return buffer_to_image(buffer=io.BytesIO())
        else:
            plt.show()
