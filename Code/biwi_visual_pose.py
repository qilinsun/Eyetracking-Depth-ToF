from getEstimation_pcd import voting,transform
import open3d as o3d
import numpy as np
from read_depth import loadDepthImageCompressed

input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/LibData/"





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
    tri.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    o3d.visualization.draw_geometries([lines,pcd,tri])

# def invert(nparray):
#     h,w = nparray.shape
#     res = np.zeros_like(nparray)
#     for i in range(h):
#         res[h-1-i] = nparray[i]
#     return res

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
    depth_npy = depth_npy[h_cut:-1,:]
    # h_cut = int(2*h/3)
    # depth_npy = depth_npy[0:h_cut,:]
    h = h - h_cut
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


# mesh = o3d.io.read_triangle_mesh(input_folder+test_face)

# pcd = o3d.geometry.PointCloud(mesh.vertices)
# pts = np.asarray(pcd.points)
# trans = np.asarray([[0.975295133269058,-7.74755070159184e-02,0.206874234341343,-9540],
# [8.15026117402656e-02,0.996612401205402,-1.10020926699975e-02,507.471524403633],
# [-0.205321034726611,2.75910778374109e-02,0.978305680818917,-4090460.44523490],
# [0,0,0,1]])
# trans = np.asarray([[0.9,-7.74755070159184e-02,0.206,0],
# [8.15026117402656e-02,0.79,-1.10020926699975e-02,0],
# [-0.205321034726611,2.75910778374109e-02,0.97,0],
# [0,0,0,1]])
# trans = np.asarray(\
# [[0.931,-0.112,0.348,0],
# [0.154,0.983,-9.62e-02,0],
# [-0.331,0.143,0.933,0],
# [0,0,0,1]])
# trans = np.asarray(\
# [[0.951,-0.112,0.1,0],
# [0.154,0.983,-9.62e-02,0],
# [-0.131,0.143,0.953,0],
# [0,0,0,1]])
# # trans = np.identity(4)
# for i in range(int(len(pts))):
#     pts[i] = transform(pts[i],trans)
# pcd.points = o3d.utility.Vector3dVector(pts)
# pcd1 = pcd.voxel_down_sample(7000)

# dis = np.linalg.norm(voting_res[0][0]-trans[0:3,0:3])
pcd = get_pcd(2)
pcd1 = pcd.voxel_down_sample(3000)
voting_res = voting(save_folder,pcd1,5,1,tlr = 3000)
visualize(voting_res,pcd)

