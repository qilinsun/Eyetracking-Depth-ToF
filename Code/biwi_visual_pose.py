from getEstimation_pcd import voting,transform
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from read_depth import loadDepthImageCompressed
import matplotlib.pyplot as plt
from edge import detect_pupil
import cv2 as cv


input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/LibData/"


# Orientations, estimated centroids and landmarks, and sample triangles
def visualize(res,pcd, plane_pupils):
    _,nose,nose_tip,left_eye,right_eye = res[1]
    R = res[0]
    print(R)
    trans = np.identity(4)
    trans[:3,:3] = R
    trans[3,3] = 1
    orie_vec = transform(np.asarray([0,0,1]),trans)
    orie_vec *= 100000
    vertices = np.array(pcd.points)
    left_cen = left_eye - transform(np.asarray([0,0,12000]),trans)
    right_cen = right_eye - transform(np.asarray([0,0,12000]),trans)
    real_left, real_right = get_3D_pupils(plane_pupils,res)
    print(left_eye,left_cen,real_left)
    # For drawing
    left_end = 5 * np.asarray(real_left) - 4 * left_cen
    right_end = 5 * np.asarray(real_right) - 4 * right_cen    
    # left_end = np.asarray(real_left)
    # right_end = np.asarray(real_right)
    print("Start visualizing...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.array([nose,nose_tip,real_left,real_right,\
                                                        nose+orie_vec/2,nose_tip+orie_vec,left_end,right_end]))
    lines.lines = o3d.utility.Vector2iVector(np.array([[0,4],[1,5],[2,6],[3,7]]))
    lines.colors = o3d.utility.Vector3dVector(np.array([[1,0.5,0],[0,0,1],[0,0,0],[0,0,0]]))
    o3d.visualization.draw_geometries([lines,pcd])
    o3d.io.write_point_cloud("./temp/face.ply",pcd)

def read_camera_para(cal):
    intrin = np.zeros((3,3),np.float32)
    rot = np.zeros((3,3),np.float32)
    translation = np.zeros((3,),np.float32)
    with open(cal) as f:
        for i in range(3):
            temp = f.readline().split(' ')
            intrin[i,:] = temp[0:3]
        for i in range(3):
            f.readline()
        for i in range(3):
            temp = f.readline().split(' ')
            rot[i,:] = temp[0:3]
        f.readline()
        temp = f.readline().split(' ')
        translation = np.asarray(temp[0:3])
    return intrin,rot,translation


def get_pcd(number):
    number = str(number)
    intrinsic,rot,translation = read_camera_para(\
        "./data/input/"+number+"/depth.cal")
    rgbd_folder = "./data/input/" + number + "/"
    depth_npy = loadDepthImageCompressed(rgbd_folder + number + ".bin")
    h, w = depth_npy.shape
    depth_npy = np.flipud(depth_npy)
    h_cut = int(h/2.8)
    depth_npy[0:h_cut,:] = 0
    # h_cut = int(2*h/3)
    # depth_npy = depth_npy[0:h_cut,:]
    # h = h - h_cut
    depth_npy = np.array(depth_npy,dtype = np.float32)
    depth_scale_factor = 1.0 / 1000.0
    depth_npy *= depth_scale_factor
    depth_raw = o3d.geometry.Image(depth_npy)
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[0:3, 0:3] = rot
    extrinsic_matrix[0:3, 3] = translation
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,\
                            o3d.camera.PinholeCameraIntrinsic(w,h,intrinsic),depth_scale = 1.0)
                                # extrinsic=extrinsic_matrix)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000000)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])
    return pcd

# R, landmarks: nasal bridge, tip, left and right pupils
def get_pupil(pcd,res):
    neigh = NearestNeighbors(n_neighbors=1,radius=2000)
    neigh.fit(np.asarray(pcd.points))
    left, right = res[1][2:4]
    left_ind,right_ind= neigh.kneighbors([left,right],1,False)
    return left_ind[0],right_ind[0]

def RGB2YUV(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    # Your code here
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    U = -0.148*R - 0.291*G +0.439*B + 128
    V = 0.439*R - 0.368*G -0.072*B + 128
    print("Y value: ", Y.max())
    return np.stack([Y,U,V], axis=2)

def RGB2Gray(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Gray = 0.2989*R + 0.5870*G + 0.1140*B
    return Gray

# For debug only
def get_pcd_2d(pcd):
    pts = np.array(pcd.points)/1000
    trans = np.asarray([[575.816,0,320],
                        [0,575.816,240],
                        [0,0,1]])
    new_pts = np.dot(trans,pts.T).T
    ret_pts = np.zeros((480,640),np.float32)
    for i in range(new_pts.shape[0]):
        new_pts[i][0:2] /= new_pts[i][2]
    for i in range(new_pts.shape[0]):
        ret_pts[480 - round(new_pts[i][1]),round(new_pts[i][0])] = new_pts[i][2]
    return ret_pts

# Default Intrinsic:
# 575.816 0 320 
# 0 575.816 240 
# 0 0 1 
FX = 575.816
FY = FX
CX = 320
CY = 240

# Deviation(pixles) when projecting the 3D positions to 2D image.
DEV = 13

# Project the 3D location of pupils to the 2D image.
# Remember to modify the intrinsics if using a different set of images!
# Parameters: index of the image, number of clusters during voting
def visualize_pupil(num, voting_res):
    rgb =  plt.imread("./data/input/" + str(num) + "/" + str(num) + ".png") * 255
    gray = RGB2Gray(rgb)
    intensity = cv.GaussianBlur(gray,(5,5),0.005*gray.shape[0])
    left, right = voting_res[1][3:5]
    left_p = left / 1000
    right_p = right / 1000
    # The offsets are fixed and are due to the fact that
    # the intensity image that open3d gets have deviations from the real one.
    x1 = round(left_p[0] * FX / left_p[2] + CX) - DEV
    y1 = 480 - round(left_p[1] * FY / left_p[2] + CY) + DEV
    x2 = round(right_p[0] * FX / right_p[2] + CX) - DEV
    y2 = 480 - round(right_p[1] * FY / right_p[2] + CY) + DEV
    R = voting_res[0]
    print(R)
    print([x1,y1])
    print([x2,y2])
    # print(left_p,right_p)
    # print(([x1,x2],[y1,y2]))
    # trans = np.identity(4)
    # trans[:3,:3] = R
    # trans[3,3] = 1
    # orie_vec = transform(np.asarray([0,0,1]),trans)
    # orie_vec *= 50000
    # plt.figure()
    # plt.plot([x1,x2],[y1,y2],color = (1,0,1))
    # plt.imshow(get_pcd_2d(pcd))
    # plt.show()
    # plt.figure()
    # plt.imshow(intensity)
    # plt.plot([x1,x2],[y1,y2],color = (1,0,1))
    # plt.show()
    return ([x1,y1],[x2,y2]), intensity

def get_3D_pupils(pupils, voting_res):
    left, right = voting_res[1][3:5]
    print("Pupils: ",pupils)
    print("Left & Right: ", left, right)
    # x and y are exchanged during 2-D image processing, so here
    # we exchange them back.
    x1 = (pupils[0][1] + DEV - CX) * (left[2]) / FX
    y1 = (480 - pupils[0][0] + DEV - CY) * (left[2]) / FY
    x2 = (pupils[1][1] + DEV - CX) * (right[2]) / FX
    y2 = (480 - pupils[1][0] + DEV - CY) * (right[2]) / FY
    return [x1,y1,left[2]],[x2,y2,right[2]]

# pcd = get_pcd(2)
# pcd1 = pcd.voxel_down_sample(3000)
# voting_res = voting(save_folder,pcd1,5,1,cluster_num = 100, tlr = 3000)
if __name__ == "__main__":
    num = 4
    # Parameters: index of the image, times of voting
    pcd = get_pcd(num)
    pcd1 = pcd.voxel_down_sample(3000)
    vt_res = voting(save_folder,pcd1,5,1,cluster_num = 100, tlr = 5000)
    img = cv.imread(input_folder+str(num)+"/"+str(num)+".png",cv.IMREAD_GRAYSCALE)
    temp = visualize_pupil(num,vt_res)
    image_pupil = detect_pupil(temp[1],temp[0])
    left, right = image_pupil
    plt.figure()
    plt.imshow(temp[1])
    plt.plot([temp[0][0][0],temp[0][1][0]],[temp[0][0][1],temp[0][1][1]],color = (1,0,1))
    plt.plot([left[1],right[1]],[left[0],right[0]],color = (1,1,1))
    plt.savefig("./temp/result.png")
    visualize(vt_res,pcd,image_pupil)
    # left, right = detect_pupil(temp[1],temp[0])
    # print(left,right)
    # intensity = temp[1]
    # # rgb =  plt.imread("./data/input/" + str(num) + "/" + str(num) + ".png") * 255
    # # intensity = RGB2YUV(rgb)[:,:,0]
    # plt.imshow(intensity)
    # # Pink Line: Original
    # # White Line: Estimated Location
    





