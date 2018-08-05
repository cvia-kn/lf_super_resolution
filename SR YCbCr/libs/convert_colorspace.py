import numpy as np
from scipy import linalg
from skimage import img_as_float, dtype_limits


def rgb2YCbCr(img):

    ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966],
                               [-37.797, -74.203, 112.0],
                               [112.0, -93.786, -18.214]])/255

    img_y = ycbcr_from_rgb[0, 0] * img[:,:,0] + \
            ycbcr_from_rgb[0, 1] * img[:,:,1] + \
            ycbcr_from_rgb[0, 2] * img[:,:,2] +16/255
    img_cb = ycbcr_from_rgb[1, 0] * img[:,:,0] + \
             ycbcr_from_rgb[1, 1] * img[:,:,1] + \
             ycbcr_from_rgb[1, 2] * img[:,:,2] + 128/255
    img_cr = ycbcr_from_rgb[2, 0] * img[:,:,0] + \
             ycbcr_from_rgb[2, 1] * img[:,:,1] + \
             ycbcr_from_rgb[2, 2] * img[:,:,2] + 128/255

    img_ycbcr = np.zeros(img.shape)
    img_ycbcr[:, :, 0] = img_y
    img_ycbcr[:, :, 1] = img_cb
    img_ycbcr[:, :, 2] = img_cr

    return img_ycbcr


def YCbCr2rgb(img):
    img_rgb = np.zeros(img.shape)

    ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966],
                               [-37.797, -74.203, 112.0],
                               [112.0, -93.786, -18.214]])

    inverse = np.linalg.inv(ycbcr_from_rgb)

    rgb_from_ycbcr = inverse * 255

    offset = np.array([[16], [128], [128]])
    offset = np.dot(inverse, offset)

    img_rgb[:, :, 0] = rgb_from_ycbcr[0, 0] * (img[:, :, 0]) + \
                       rgb_from_ycbcr[0, 1] * (img[:, :, 1]) + \
                       rgb_from_ycbcr[0, 2] * (img[:, :, 2]) - offset[0]

    img_rgb[:, :, 1] = rgb_from_ycbcr[1, 0] * (img[:, :, 0]) + \
                       rgb_from_ycbcr[1, 1] * (img[:, :, 1]) + \
                       rgb_from_ycbcr[1, 2] * (img[:, :, 2]) - offset[1]

    img_rgb[:, :, 2] = rgb_from_ycbcr[2, 0] * (img[:, :, 0]) + \
                       rgb_from_ycbcr[2, 1] * (img[:, :, 1]) + \
                       rgb_from_ycbcr[2, 2] * (img[:, :, 2]) - offset[2]

    return img_rgb

