import skvideo.io
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import matplotlib.pyplot as plt

Video_Dir = './wave_2_enhanced_1.avi'
Video_Data = skvideo.io.vread(Video_Dir)
[Num_Frms, H, W, Pix_Dim] = Video_Data.shape

## frame 0
frame0 = Video_Data[0]
frame0_flattened = np.reshape(frame0, (H * W, Pix_Dim))
frame0_flattened_whitened = whiten(frame0_flattened)
centroids0, labels0 = kmeans2(frame0_flattened_whitened, 3, minit='points')
labels0 = np.reshape(labels0, (H, W))
#fig = plt.figure(figsize=(8,8))
#fig.add_subplot(121)
#plt.imshow(Video_Data[0])
#fig.add_subplot(122)
plt.imshow(labels0)
plt.show()

