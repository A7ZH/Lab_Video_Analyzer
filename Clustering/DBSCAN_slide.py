import cv2
def slide(sec):
    for i in range(0, 190):
        im = cv2.imread('DBSCAN_imgs/crop_image_'+str(i)+'.png')
        cv2.imshow('frame '+str(i), im)
        cv2.waitKey(sec)
