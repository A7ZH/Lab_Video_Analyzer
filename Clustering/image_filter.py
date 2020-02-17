import numpy as np
import sklearn
from scipy.cluster.vq import kmeans2, whiten
import cv2,os
import matplotlib.pyplot as plt
import glob as gb
from scipy.ndimage import gaussian_filter
import scipy.misc
import math
from scipy import signal
from scipy.stats import entropy
import mpl_toolkits.mplot3d as p3d

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def kl_divergence(p, q):
    # return a number, entropy function has normalized matrices
    return np.sum(entropy(p,q))


def conv2d(my_img,kernel,size=5,skip_size=1):
    img_shape = my_img.shape
    margin = math.floor(size/2)
    new_img = np.zeros(img_shape,dtype="float")
    # size is odd:
    if size %2 != 0:
        margin = math.floor(size / 2)
        for i in range(margin,img_shape[0]-margin):
            for j in range(margin,img_shape[1]-margin):
                current_matrix = my_img[i-margin:i+margin+1,j-margin:j+margin+1]
                cur_score = kl_divergence(kernel, current_matrix)
                new_img[i][j] = cur_score
                j += skip_size
            i += skip_size
    else:
        for i in range(margin,img_shape[0]-margin+1):
            for j in range(margin,img_shape[1]-margin+1):
                current_matrix = my_img[i-margin:i+margin,j-margin:j+margin]
                cur_score = kl_divergence(kernel, current_matrix)
                new_img[i][j] = cur_score
                j += skip_size
            i += skip_size
    return new_img

temp_p = np.arange(25).reshape((5,5))

my_kernel = gkern(6,std=1)
# print(my_kernel)
plt.imshow(my_kernel, interpolation='none')
plt.colorbar()
plt.show()

first_image = './crop_images/crop_image_0.png'
img = cv2.imread(first_image, 0)

new_img = conv2d(img,my_kernel,6)
print(new_img.shape)
# new_img[np.where(new_img == np.nan)] = 0.
new_img = np.nan_to_num(new_img)
new_img[new_img > 100.] = 0.
new_img[new_img < 6.] = 0.
print(new_img[0:100,0:100])

print(np.amax(new_img))

plt.imshow(new_img, interpolation='none')
plt.colorbar()
plt.show()
plt.imshow(img, interpolation='none')
plt.colorbar()
plt.show()






