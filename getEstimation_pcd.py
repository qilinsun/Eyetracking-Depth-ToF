import numpy as np
import open3d as o3d
from train_flann import ICPtransform, getDescriptor
from random import randint
from sklearn.neighbors import NearestNeighbors
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

def getNormals(pcd):
    # o3d.visualization.draw_geometries([pcd])
    downpcd = pcd.voxel_down_sample(voxel_size=0.001)
    # o3d.visualization.draw_geometries([downpcd])
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30000, max_nn=30))
    return downpcd.normals

def rotation(v,n,theta):
    theta = theta * math.pi/180
    ret = v*np.cos(theta) + np.cross(n,v)*np.sin(theta) + \
        np.dot(n,v)*n*(1-np.cos(theta))
    return ret

def sampleFromVertices(pcd,l):
    normal_set = getNormals(pcd)
    vertices = np.array(pcd.points)
    index = randint(0,len(normal_set)-1)
    point = vertices[index]
    normal = normal_set[index]
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

def transform(vec,trans,is_point = 0):
    target = np.dot(trans[:3,:3],vec.T).T
    target += trans[:3,3] * is_point
    return target

# For Debug
def visual_tri(tri_list):
    triangles = []
    for i in tri_list:
        tri = o3d.geometry.LineSet()
        tri.points = o3d.utility.Vector3dVector(i)
        tri.lines = o3d.utility.Vector2iVector(np.array([[0,1],[1,2],[0,2]]))
        triangles.append(tri)
    # Target, before, after
    triangles[0].colors = o3d.utility.Vector3dVector(np.array([[0,0,0],[0,0,0],[0,0,0]]))
    triangles[1].colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[1,0,0],[1,0,0]]))
    # triangles[2].colors = o3d.utility.Vector3dVector(np.array([[1,0.5,0],[1,0.5,0],[1,0.5,0]]))
    o3d.visualization.draw_geometries(triangles)

def vote(vectors,triangles,ind,tri,h,landmk_index):
    # ind,dis= getNearDes(des,descriptors,h)
    # # ind = ind[0]
    # print(ind,dis)
    mul_trans = []
    mul_cen = []
    mul_vec = []
    # print("IND: ",ind)
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
        # print("sample: ", B)
        # print("Corres: ", A)
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        A_centered = A - centroid_A
        B_centered = B - centroid_B
        M = np.dot(B_centered.T, A_centered)
        U, _, Vt = np.linalg.svd(M)
        # R = np.dot(Vt.T, U.T)
        R = np.dot(U,Vt)
#         R = np.asarray(\
# [[1,0,0],
# [0,0.8,0],
# [0,0,0.8]])
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(U,Vt)
#         R = np.array([[0.980977,0.0777566,0.177872], 
# [-0.0431257,-0.98067,-0.190858],
# [-0.189274,0.179556,-0.965368]])
#         R = np.array([[0.842508,-0.176389,0.508987],
# [0.00292723,0.946356,0.323114], 
# [-0.538676,-0.270736,0.797828]])
        t = centroid_B - np.dot(R, centroid_A.T).T
        # print("Distance before: ", np.sqrt(np.sum((B_centered-A_centered) ** 2)))
        # print("Distance after: ", np.sqrt(np.sum((np.dot(R,A_centered.T).T - B_centered) ** 2)))
        # visual_tri([B_centered,np.dot(R,A_centered.T).T])
        trans = np.identity(4)
        trans[:3, :3] = R
        
        trans[:3, 3] = t
        # homo = np.identity(4)
        # homo[:3,:3] = A
        # visual_tri([B,np.dot(R,A.T).T+t])
        # print("Sample Triangle: ",A_centered)
        # print("Corres before trans: ",B_centered)
        # print("Corres after trans: ",np.dot(R,B_centered))
        # print("Orientation: ",R)
        # left_pupil = np.dot(trans,np.array(corres_vec[3],np.int64).T).T + base_cen
        # print("Estimated Location: ",left_pupil)
        landmarks = []
        for j in range(landmk_index):
            landmarks.append(transform(corres_vec[j],trans,1))
        centroid = transform(corres_vec[0],trans,1)
        mul_trans.append(R)
        mul_cen.append(centroid)
        mul_vec.append(landmarks)
    return mul_trans,mul_cen,mul_vec


def multi_vote(n,flann,pcd,vectors,triangles,tolerance,landmk_index=5,h=1,l=80000, d=3000, k=5):
    orients = []
    centroids = []
    landmarks = []
    i = 0
    # vertices = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000000)
    vertices = pcd.points
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vertices)
    while (i < n):
        # try:
        tri, distances = ICPtransform(\
                vertices,neigh,sampleFromVertices(pcd,l))
        if (distances[0]>d or distances[1]>d or distances[2]>d):
            # print(distances)
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
        if (dis > tolerance):
            print(dis)
            continue
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
# mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
def voting(save_folder,pcd,landmk_index,h,tlr = 5000):
    print("Start voting...")
    with open(save_folder+'Descriptors.npy', 'rb') as f:
        descriptors = np.load(f)
    flann = pyflann.FLANN()
    flann.load_index(save_folder+'Index',descriptors)
    with open(save_folder+'Vectors.npy', 'rb') as f:
        vectors = np.load(f) # vec: v_c,v_1,v_2,u_1,u_2
    with open(save_folder+'Triangles.npy', 'rb') as f:
        triangles = np.load(f)

    t = time.time()

    # orients,centroids,landmarks, samples
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    votes = multi_vote(1,flann,pcd,vectors,triangles,tlr,landmk_index, h)
    # print("Real location: ",vertices[4280])
    print("Voting Time: ",time.time()-t)
    return votes


# print(vertices[12278])
# p1 = vertices[12278] - vertices[4280] # base
# p2 = vectors[0][5]-vectors[0][4] # corres
# print(p1)
# print(p2)





