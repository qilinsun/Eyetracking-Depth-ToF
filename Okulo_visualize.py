from getEstimation_pcd import voting,transform
import open3d as o3d
import numpy as np
import cv2 as cv
from read_depth import loadDepthImageCompressed

input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/PositionLib/data/"


# Orientations, estimated centroids and landmarks, and sample triangles
def visualize(res,pcd):
    _,nose,nose_tip,right_eye,left_eye = res[2][0]
    R = res[0][0]
#     R = np.asarray([[0.778027,-0.111741,0.618213], 
# [-0.0218606,-0.978643,0.2044], 
# [-0.62785,-0.172543,-0.758968] ])
    print(R)
    # orie_vec = np.dot(R,np.array([0,0,1]))
    trans = np.identity(4)
    trans[:3,:3] = R
    trans[3,3] = 1
    orie_vec = transform(np.asarray([0,0,1]),trans)
    orie_vec *= 100000
    # print("Orien vector: ", orie_vec)
    vertices = np.array(pcd.points)
    print("Start visualizing...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    # lines = o3d.geometry.LineSet()
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.array([nose,nose_tip,left_eye,right_eye,\
                                                        nose+orie_vec/2,nose_tip+orie_vec,left_eye+orie_vec/2,right_eye+orie_vec/2]))
    lines.lines = o3d.utility.Vector2iVector(np.array([[0,4],[1,5],[2,6],[3,7]]))
    lines.colors = o3d.utility.Vector3dVector(np.array([[1,0.5,0],[0,0,1],[0,0,0],[0,0,0]]))
    tri = o3d.geometry.LineSet()
    tri.points = o3d.utility.Vector3dVector(res[3][0])
    tri.lines = o3d.utility.Vector2iVector(np.array([[0,1],[1,2],[0,2]]))
    tri.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[1,0,0],[1,0,0]]))
    o3d.visualization.draw_geometries([lines,pcd,tri])

# def invert(nparray):
#     h,w = nparray.shape
#     res = np.zeros_like(nparray)
#     for i in range(h):
#         res[h-1-i] = nparray[i]
#     return res


def get_pcd():
    intrinsic = np.asarray([[530.325317, 0.000000, 321.097351],
   [0.000000, 530.411377, 246.448624],
   [0.000000, 0.000000, 1.000000]])
    rot = np.identity(3)
    translation = np.zeros((3,),np.float32)
    file = input_folder + "Okulo/"
    depth_npy = cv.imread(file + "2.pgm")[:,:,0] # Value: 0~187, unit: m
    mask = depth_npy < 100
    depth_npy = depth_npy * mask
    h, w = depth_npy.shape
    depth_npy = np.flipud(depth_npy)
    # h_cut = int(h/3)
    # depth_npy = depth_npy[h_cut:-1,:]
    depth_npy = np.array(depth_npy,dtype = np.float32)
    # depth_scale_factor = 1.0 / 1000.0
    # depth_npy *= depth_scale_factor
    depth_raw = o3d.geometry.Image(depth_npy)
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[0:3, 0:3] = rot
    extrinsic_matrix[0:3, 3] = translation
    
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,\
                            o3d.camera.PinholeCameraIntrinsic(w,h,intrinsic),depth_scale = 1.0)
                                # extrinsic=extrinsic_matrix)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)*5000)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])
    return pcd

pcd = get_pcd()
pcd1 = pcd.voxel_down_sample(3000)
# o3d.visualization.draw_geometries([pcd])

voting_res = voting(save_folder,pcd,5,1,tlr = 9000)
visualize(voting_res,pcd)

