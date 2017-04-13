# -*- coding: utf-8 -*-
"""
Module to convert a hips image into a material map.

The functions contained in this module were developed by Phil Wilkes
(p.wilkes@ucl.ac.uk) at the Dept. of Geography, University College London.


"""

import os
import numpy as np
import matplotlib.pyplot as plt


def read_header(fname):

    """
    :param fname:
    :return:

    header_length, bands, res_x, res_y, fmt
    """

    # grab header info
    header_length = open(fname).read().find('\n.')
    header = open(fname).read()[:header_length].split()
    bands = int(header[1])
    res_x = int(header[2])
    res_y = int(header[3])
    fmt = int(header[4])

    return header_length, bands, res_x, res_y, fmt


def read_hips(fname):

    header_length, bands, res_x, res_y, fmt = read_header(fname)

    # extract image from hips
    hips = np.fromfile(fname, np.uint8)
    hips_length = len(hips)
    img = hips[hips_length - (bands * res_x * res_y):].reshape(bands, res_x, res_y)
    img = np.rot90(img, 2)

    return img[0, :, :], res_x, res_y, fmt


def hips2img(fname, imshow=True, imsave=False):

    img, res_x, res_y, fmt = read_hips(fname)

    if imshow:
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.axis('off')

    # save image
    if imsave:
        plt.imsave(os.path.splitext(fname)[0] + '.png', img,
                   cmap='gray')
