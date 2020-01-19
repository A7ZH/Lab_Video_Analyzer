import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

###################### READ_IN_IMAGE ###########################################
                                                        # grayscale
# pic = np.array(cv2.imread('DBSCAN_imgs/crop_image_94.png', 0), dtype='float')
pic = np.array(plt.imread('Frame1.png'), dtype='float')[172:773, 292:1055]
def rgba2gray(rgba):
  return np.dot(rgba[...,:4], [0.2989, 0.5870, 0.1140,0])
pic = rgba2gray(pic)
plt.subplot(2,2,1)
plt.imshow(pic, cmap='gray')

###################### CLUSTERING ALL PIXELS BY COLOR#######################
H, W = pic.shape
pic_flattened = np.reshape(pic, (H*W))
color_centroids, color_labels = kmeans2(pic_flattened, 2, minit='points')
color_labels_pic = np.reshape(color_labels,(H,W))
plt.subplot(2,2,2)
plt.imshow(color_labels_pic, cmap='gray')

################  CLUSTERING PARTICLE PIXELS BY EUCLIDEAN DISTANCE #########
#extract nonbackground pixels; backgound pixels labelled 1
X, Y = np.where(color_labels_pic!=1)
plt.subplot(2,2,3)
plt.scatter(X,Y)
particle_coords = np.array(np.vstack((X,Y)).T, dtype='float')
particle_coords_centroids, particle_coords_labels = kmeans2(particle_coords, 8, minit='points')
plt.subplot(2,2,4)
plt.scatter(particle_coords_centroids[:,0], particle_coords_centroids[:, 1])
plt.show()
