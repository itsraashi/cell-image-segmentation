# https://medium.com/analytics-vidhya/detecting-and-counting-objects-with-opencv-b0f59bc1e111
import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread('LiveDead Image Analysis/MFGTMP_210413160001_A01f03d0.PNG')
cv.imwrite('1.original_image.png', image)
k = np.array([[1, 0, 1],
                   [0, 3, 0],
                   [1, 0, 1]])
image_sharp = cv.filter2D(src=image, ddepth=-1, kernel=k)
cv.imwrite('2.sharpen_image.png', image_sharp)
image_blur_gray = cv.cvtColor(image_sharp, cv.COLOR_BGR2GRAY)
image_res ,image_thresh = cv.threshold(image_blur_gray,130,255,cv.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(image_thresh,cv.MORPH_OPEN,kernel) 
cv.imwrite('3.set_threshold.png', opening)

'''
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, last_image =  cv.threshold(dist_transform, 0.3*dist_transform.max(),255,0)
last_image = np.uint8(last_image)
cv.imwrite('4.transform_image.png', last_image)
'''

cnts = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


def display(img,count,cmap="gray"):
    f_image = cv.imread('LiveDead Image Analysis/MFGTMP_210413160001_A01f03d0.PNG')
    f, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].imshow(f_image,cmap="gray")
    axs[1].imshow(img,cmap="gray")
    axs[1].set_title("Total Count = {}".format(count))

for (i, c) in enumerate(cnts):
	((x, y), _) = cv.minEnclosingCircle(c)
	#cv.putText(image, "#{}".format(i + 1), (int(x) - 45, int(y)+20),
		#cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
	cv.drawContours(image, [c], -1, (0, 255, 0), 2)
cv.imwrite('5.find_contours.png', image)

display(image, len(cnts))

print(len(cnts))

