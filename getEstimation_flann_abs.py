import numpy as np
import open3d as o3d
from train_flann_abs import ICPtransform, getDescriptor
from random import randint
from sklearn.neighbors import KDTree
import math
import pyflann
import time


# Path of the images
input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/PositionLib/data/"
# save_folder = "./data/library/test/"
# depth_name = ""

# Transform a depth image to point cloud with provided intrinsic of the camera
# sintrin = o3d.camera.PinholeCameraIntrinsic
# pointCloud = o3d.geometry.PointCloud.create_from_depth_image(\
    # depth_image,intrin)
# print(pointCloud)

def getNormals(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    # o3d.visualization.draw_geometries([pcd])
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # o3d.visualization.draw_geometries([downpcd])
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30000, max_nn=30))
    return downpcd.normals

def rotation(v,n,theta):
    theta = theta * math.pi/180
    ret = v*np.cos(theta) + np.cross(n,v)*np.sin(theta) + \
        np.dot(n,v)*n*(1-np.cos(theta))
    return ret

def sampleFromVertices(vertices,l):
    index = randint(0,len(vertices)-1)
    point = vertices[index]
    normal = getNormals(vertices)[index]
    r = np.random.random(3)
    r_parallel = r - np.dot(normal,r)*normal
    v1 = r_parallel/np.sqrt(np.dot(r_parallel,r_parallel)) * l / 1.732
    v2 = rotation(v1,normal,120)
    v3 = rotation(v1,normal,240)
    eqTriangle = np.array((point+v1,point+v2,point+v3))
    return eqTriangle



# Read from the library.
# Descriptors: 2-d array of descriptors of base triangles
# stored in KDTree. 
# Vec: tri,v_c,v_1,v_2,u_1,u_2
# namely. base triangle, and the vectors starting from center of the tri to:
# center of head, deepest point of nasal bridge, nose tip, left pupil, right pupil
# The indices of descriptors and vectors are corresponding


def getNearDes(des,fnn,h):
    # s = pickle.dumps(des_lib)
    # tree_copy = pickle.loads(s)
    # dis,ind = tree_copy.query(des.reshape(1,-1),1)
    # flann = pyflann.FLANN()
    # testset = np.array([des])
    # ind,dis = flann.nn(des_lib,testset,h,algorithm="kmeans",branching = 32, iterations = 10, checks = 16)
    des = np.array(des,dtype = np.int32)
    ind,dis = fnn.nn_index(des,1,checks = 1000)
    dis = np.sqrt(dis)
    return ind,dis

# mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
# vertices = np.asarray(mesh.vertices)
# s = sampleFromVertices(vertices,l = 80000)
# tri, distances= ICPtransform(vertices,s)
# d = 3000
# while (distances[0]>d or distances[1]>d or distances[2]>d):
#         tri,distances = ICPtransform(vertices,sampleFromVertices(vertices,l = 80000))
# des = getDescriptor(vertices,tri, 80000, 5)

def transform(point,trans):
    point_homo = np.append(point,1)
    target_homo = np.dot(trans,point_homo)
    target = target_homo[:3] / target_homo[3]
    return target

def vote(vectors,triangles,ind,tri,h,landmk_index):
    # ind,dis= getNearDes(des,descriptors,h)
    # # ind = ind[0]
    # print(ind,dis)
    mul_trans = []
    mul_cen = []
    mul_vec = []
    for i in range(len(ind)):
        index = ind[i]
        corres_vec = vectors[index]
        corres_tri = triangles[index]
        base_cen = (tri[0]+tri[1]+tri[2])/3
        # base_cen_tri = tri - base_cen
        corres_cen = (corres_tri[0] + corres_tri[1] + corres_tri[2]) /3
        # corres_cen_tri = corres_tri - corres_cen
        # print("tri: ",tri)
        # print("corres: ",corres_tri)
        # print("Corres_Des: ",getDescriptor(vertices,corres_tri,80000,5))
        # trans = np.dot(base_cen_tri.T,np.linalg.inv(corres_cen_tri.T))
        # left_pupil = np.dot(trans,corres_cen_tri.T).T
        # print(left_pupil)
        A = np.array(corres_tri,np.int64)
        B = np.array(tri,np.int64)
        # print("tricen: ",a)
        # print("correscen: ",b)
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        A_centered = A - centroid_A
        B_centered = B - centroid_B
        M = np.dot(B_centered.T, A_centered)
        U, _, Vt = np.linalg.svd(M)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_B - np.dot(R, centroid_A)
        trans = np.identity(4)
        trans[:3, :3] = R
        trans[:3, 3] = t
        # print("Orientation: ",R)
        corres_pupil = corres_vec[3]
        # left_pupil = np.dot(trans,np.array(corres_vec[3],np.int64).T).T + base_cen
        left_pupil = np.dot(R,corres_pupil) + t
        # print("Estimated Location: ",left_pupil)
        landmarks = []
        for j in range(landmk_index):
            landmarks.append(transform(corres_vec[j],trans))
        centroid = transform(corres_vec[0],trans)
        mul_trans.append(R)
        mul_cen.append(centroid)
        mul_vec.append(landmarks)
    return mul_trans,mul_cen,mul_vec
    
def multi_vote(n,flann,vertices,vectors,triangles,landmk_index=5,h=1,l=80000, d=3000, k=5):
    orients = []
    centroids = []
    landmarks = []
    i = 0
    while (i < n):
        # try:
        tri, distances = ICPtransform(\
                vertices,sampleFromVertices(vertices,l))
        if (distances[0]>d or distances[1]>d or distances[2]>d):
            continue
        # index = 5
        # tri = triangles[index]+1000*np.random.random((1,3))
        # tri = triangles[index]
        des = getDescriptor(vertices,tri, l, k)
        # print("Des: ",des)
        ind,dis= getNearDes(des,flann,h)
        # 5000: bad
        # 3000: good : bad = 6 : 4
        # 2000: often good, but too slow
        if (dis > 2000):
            continue
        # ind = ind[0]
        print(ind,dis)
        samples = []
        ret = vote(vectors,triangles,ind,tri,h,landmk_index)
        for j in range(len(ret[0])):
            orients.append(ret[0][j])
            centroids.append(ret[1][j])
            landmarks.append(ret[2][j])
            samples.append(tri)
        i += 1
        # except:
        #     continue
    if landmk_index == -1:
        return orients,centroids,[]
    return orients,centroids,landmarks,samples


def isNear(R1, R2, theta):
    fdis = np.linalg.norm(R1-R2)
    print(fdis)
    return (fdis < 2.828*abs(math.sin(theta/360*math.pi)))

# return 4 arrays: datas from multiple votings
# Orientations, estimated centroids and landmarks(0,1,2,3,4), and sample triangles
mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
def voting(save_folder,mesh,landmk_index,h):
    print("Start voting...")
    with open(save_folder+'Descriptors.npy', 'rb') as f:
        descriptors = np.load(f) # 目前只存了5000个triangle...
    flann = pyflann.FLANN()
    flann.load_index(save_folder+'Index',descriptors)
    with open(save_folder+'Vectors.npy', 'rb') as f:
        vectors = np.load(f) # vec: v_c,v_1,v_2,u_1,u_2
    with open(save_folder+'Triangles.npy', 'rb') as f:
        triangles = np.load(f)

    t = time.time()
    vertices = np.asarray(mesh.vertices)

    # orients,centroids,landmarks, samples
    votes = multi_vote(1,flann,vertices,vectors,triangles,landmk_index,h)
    # print("Real location: ",vertices[4280])
    print("Voting Time: ",time.time()-t)
    return votes
# print(vertices[12278])
# p1 = vertices[12278] - vertices[4280] # base
# p2 = vectors[0][5]-vectors[0][4] # corres
# print(p1)
# print(p2)





