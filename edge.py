import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

number = 2
input_folder = "./data/input/"

img = cv.imread(input_folder+str(number)+"/"+str(number)+".png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"



_,img_1 = cv.threshold(img,100,200,cv.THRESH_BINARY)

img_1 = cv.Canny(img,75,150,3)


print("Show Image...")
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_1,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()