import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# from biwi_visual_pose import visualize_pupil
# number = "Okulo"

input_folder = "./data/input/"
num = 2
img = cv.imread(input_folder+str(num)+"/"+str(num)+".png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reference: https://thume.ca/projects/
# 2012/11/04/simple-accurate-eye-center-tracking-in-opencv/
def compute_mat_x_gradient(mat):
    out = np.zeros((mat.shape[0], mat.shape[1]), dtype=np.float64)
    for y in range(mat.shape[0]):
        Mr = mat[y, :]
        Or = out[y, :]
        Or[0] = Mr[1] - Mr[0]
        for x in range(1, mat.shape[1] - 1):
            Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0
        Or[mat.shape[1] - 1] = Mr[mat.shape[1] - 1] - Mr[mat.shape[1] - 2]
    return out

def compute_mat_y_gradient(mat):
    return compute_mat_x_gradient(mat.T).T

def get_grad_map(zone):
    dev = 0
    mean = np.mean(zone)
    mask = np.zeros_like(zone)
    for i in range(zone.shape[0]):
        for j in range(zone.shape[1]):
            dev += (zone[i,j] - mean)**2
    dev = (dev/(zone.shape[0]*zone.shape[1]))**0.5
    thres = mean - 0.3*dev
    print(mean,dev)
    x_grad = compute_mat_x_gradient(zone)
    y_grad = compute_mat_y_gradient(zone)
    mag = np.zeros_like(zone)
    for i in range(zone.shape[0]):
        for j in range(zone.shape[1]):
            temp = (x_grad[i,j]**2 + y_grad[i,j]**2)**0.5
            mag[i,j] = temp
            mask[i,j] = temp <= thres
    plt.imshow(mask*x_grad)
    plt.savefig("./temp/X_grad.png")
    plt.imshow(mask*y_grad)
    plt.savefig("./temp/Y_grad.png")
    return mask*x_grad,mask*y_grad,mask*mag

def locate(zone,is_left = 0):
    # grad_map = get_grad_map(zone)
    # pos = np.asarray((-1,-1))
    # sum_map = np.zeros_like(zone)
    # for i in range(zone.shape[0]):
    #     for j in range(zone.shape[1]):
    #         sum = 0
    #         count = 0
    #         for x in range(max(0,i-5),min(zone.shape[0],i+5)):
    #             for y in range(max(0,j-5),min(zone.shape[1],j+5)):
    #                 mag = grad_map[2][x,y]
    #                 if mag > 0:
    #                     count += 1
    #                     length = ((x-i)**2+(y-j)**2)**0.5
    #                     vec_x = (x-i)*grad_map[0][x,y]
    #                     vec_y = (y-j)*grad_map[1][x,y]
    #                     if (vec_x + vec_y) > 0:
    #                         sum += ((vec_x+vec_y)/(length*mag))**2
    #         assert count != 0, "Failed to find the center!"
    #         sum /= count
    #         sum *= (255-zone[i][j])
    #         sum_map[i][j] = sum
    # pos = np.argmax(sum_map)
    # pos = np.unravel_index(pos, sum_map.shape)
    # plt.imshow(sum_map)
    # plt.plot([0,pos[1]],[0,pos[0]])
    # if is_left:
    #     plt.savefig("./temp/Left.png")
    #     plt.imshow(zone)
    #     plt.savefig("./temp/Left_Zone.png")
    # else:
    #     plt.savefig("./temp/Right.png")
    #     plt.imshow(zone)
    #     plt.savefig("./temp/Right_Zone.png")
    minval = 255
    for i in range(zone.shape[0]):
        for j in range(zone.shape[1]):
            if zone[i,j] < minval:
                pos = np.asarray((i,j))
                minval = zone[i,j]
    plt.figure()
    plt.plot([0,pos[1]],[0,pos[0]])
    print("Darkest point: ",pos)
    if is_left:
        plt.imshow(zone)
        plt.savefig("./temp/Left_Zone.png")
    else:
        plt.imshow(zone)
        plt.savefig("./temp/Right_Zone.png")
    return pos

# img.shape == (480,640)
def detect_pupil(img, pupils):
    left,right = pupils
    width = 0.4*abs(right[0] - left[0])
    height = width
    left_zone = img[int(left[1]-0.5*height):int(left[1]+0.5*height), \
                    int(left[0]-0.5*width):int(left[0]+0.5*width)]
    right_zone = img[int(right[1]-0.5*height):int(right[1]+0.5*height), \
                    int(right[0]-0.5*width):int(right[0]+0.5*width)]
    # mask = np.zeros((480,640))
    # mask[int(right[1]-0.5*height):int(right[1]+0.5*height), \
    #     int(right[0]-0.5*width):int(right[0]+0.5*width)] = 1
    # plt.imshow(left_zone)
    # plt.show()
    left_cen = locate(left_zone,1) + np.asarray((int(left[1]-0.5*height),int(left[0]-0.5*width)))
    right_cen = locate(right_zone) + np.asarray((int(right[1]-0.5*height),int(right[0]-0.5*width)))
    # plt.subplot(121),plt.imshow(left_zone,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return left_cen,right_cen

# if __name__ == "__main__":
#     detect_pupil(img,visualize_pupil(num))
# _,img_1 = cv.threshold(img,150,255,cv.THRESH_BINARY)

# img_1 = cv.Canny(img,0,70,3)

# print("Show Image...")
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_1,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()