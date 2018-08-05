import numpy as np
import matplotlib.pyplot as plt
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb
from scipy.misc import imread
from skimage.color import rgb2lab, lab2rgb
###############################
# YCbCr and RGB
###############################


# all images are default as float64
###################################
img = imread('/home/z/PycharmProjects/SR/full_data_512/platonic/input_Cam044.png')/255
img_ycbcr = rgb2YCbCr(img)
img_rgb = YCbCr2rgb(img_ycbcr)
tst = np.sum(np.abs(img - img_rgb))
print(tst)
img_err1 = (np.abs(img-img_rgb))
plt.subplot(1,3,1)
plt.imshow(img_err1[:,:,0])
plt.subplot(1,3,2)
plt.imshow(img_err1[:,:,1])
plt.subplot(1,3,3)
plt.imshow(img_err1[:,:,2])
plt.show()

# if we set the whole light field as folat32
############################################
img.astype(np.float32)
img_ycbcr = rgb2YCbCr(img).astype(np.float32) # also set the converted image as float32
img_rgb = YCbCr2rgb(img_ycbcr).astype(np.float32) # also set the converted image as float32
tst2 = np.sum(np.abs(img - img_rgb))
print(tst2)
img_err2 = (np.abs(img-img_rgb))
plt.subplot(1,3,1)
plt.imshow(img_err2[:,:,0])
plt.subplot(1,3,2)
plt.imshow(img_err2[:,:,1])
plt.subplot(1,3,3)
plt.imshow(img_err2[:,:,2])
plt.show()

# set the whole light field as folat32
###########################################
img.astype(np.float32)
img_ycbcr = rgb2YCbCr(img)
img_rgb = YCbCr2rgb(img_ycbcr).astype(np.float32)# also set the converted image as float32
tst3 = np.sum(np.abs(img - img_rgb))
print(tst3)
img_err3 = (np.abs(img-img_rgb))
plt.subplot(1,3,1)
plt.imshow(img_err3[:,:,0])
plt.subplot(1,3,2)
plt.imshow(img_err3[:,:,1])
plt.subplot(1,3,3)
plt.imshow(img_err3[:,:,2])
plt.show()

# only the whole light field as folat32
###########################################
img.astype(np.float32)
img_ycbcr = rgb2YCbCr(img)
img_rgb = YCbCr2rgb(img_ycbcr)
tst4 = np.sum(np.abs(img - img_rgb))
print(tst4)
img_err4 = (np.abs(img-img_rgb))
plt.subplot(1,3,1)
plt.imshow(img_err4[:,:,0])
plt.subplot(1,3,2)
plt.imshow(img_err4[:,:,1])
plt.subplot(1,3,3)
plt.imshow(img_err4[:,:,2])
plt.show()


# line for debugging, set breakpoint here
k = 0


##################################
# CIElab and RGB
##################################
##################################
# key point here is that the the
# img_lab must have dtype float64!
##################################
img = imread('/home/z/PycharmProjects/SR/full_data_512/platonic/input_Cam044.png')/255
img_lab =rgb2lab(img)
a = img_lab[:,:,0]
b = img_lab[:,:,1]
c = img_lab[:,:,2]
# if the dtype of lab image is float32, it gives err message
img_rgb = lab2rgb(img_lab)
# img_rgb = lab2rgb(img_lab.astype(np.float32))

plt.subplot(1,3,1)
plt.imshow(a)
plt.subplot(1,3,2)
plt.imshow(b)
plt.subplot(1,3,3)
plt.imshow(c)
plt.show()

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.show()
tst5 = np.sum(np.abs(img - img_rgb))
print(tst5)

k = 0