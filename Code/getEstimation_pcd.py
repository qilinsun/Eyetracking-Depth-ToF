import numpy as np
import open3d as o3d
from train_flann import ICPtransform, getDescriptor
from random import randint
from sklearn.neighbors import NearestNeighbors
import math
import pyflann
import time
from scipy.spatial import KDTree


# Path of the images
input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/LibData/"
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
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30000, max_nn=20))
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
    r = np.array([0,1,0])
    r = rotation(r,normal,randint(-45,45))
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

def transform(vec,trans,is_point = 0):
    target = np.dot(trans[:3,:3],vec.T).T
    target += trans[:3,3] * is_point
    return np.asarray(target)

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
    mul_trans = []
    mul_cen = []
    mul_vec = []
    index = ind[0]
    corres_vec = vectors[index]
    corres_tri = triangles[index]
    base_cen = (tri[0]+tri[1]+tri[2])/3
    corres_cen = (corres_tri[0] + corres_tri[1] + corres_tri[2]) /3
    A = np.array(corres_tri,np.int64)
    B = np.array(tri,np.int64)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    M = np.dot(B_centered.T, A_centered)
    U, _, Vt = np.linalg.svd(M)
    R = np.dot(U,Vt)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(U,Vt)
    t = centroid_B - np.dot(R, centroid_A.T).T
    trans = np.identity(4)
    trans[:3, :3] = R
    trans[:3, 3] = t
    landmarks = []
    for j in range(landmk_index):
        landmarks.append(transform(corres_vec[j],trans,1))
    centroid = transform(corres_vec[0],trans,1)
    return R,centroid,landmarks


def multi_vote(n,flann,pcd,vectors,triangles,tolerance,landmk_index=5,h=1,l=80000, d=3000, k=5):
    orients = []
    centroids = []
    landmarks = []
    i = 0
    # vertices = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000000)
    vertices = pcd.points
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vertices)
    samples = []
    while (i < n):
        # try:
        tri, distances = ICPtransform(\
                vertices,neigh,sampleFromVertices(pcd,l))
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
        if (dis > tolerance):
            continue
        # print(ind,dis)
        ret = vote(vectors,triangles,ind,tri,h,landmk_index)
        orients.append(ret[0])
        centroids.append(ret[1])
        landmarks.append(ret[2])
        samples.append(tri)
        i += 1
        # except:
        #     continue
    if landmk_index == -1:
        return orients,centroids,[]
    return orients,centroids,landmarks,samples


def isNear(vote1, vote2, theta_thres = 15, dis_thres = 25000):
    R1,c1 = vote1
    R2,c2 = vote2
    fdis = np.linalg.norm(R1-R2)
    temp = np.asarray([c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2]])
    dis = np.sqrt(np.sum(np.square(temp)))
    return (fdis < 2.828*abs(math.sin(theta_thres/360*math.pi)) and \
            (dis < dis_thres))

def cluster(voting_results):
    mark = []
    for i in range(len(voting_results[0])):
        mark.append(0)
    index = 1
    size = 0
    ret_index = 0
    while True:
        current_size = 0
        for i in range(len(mark)):
            if mark[i] == 0:
                current_size += 1
                mark[i] = index
                current = (voting_results[0][i],voting_results[1][i])
                for j in range(i,len(mark)):
                    comp = (voting_results[0][j],voting_results[1][j])
                    if (mark[j] == 0) and \
                        isNear(current,comp):
                        mark[j] = index
                        current_size += 1
                break
        if current_size == 0:
            break
        if current_size > size:
            ret_index = index
            size = current_size
        index += 1
    ret = []
    for i in range(len(mark)):
        if mark[i] == ret_index:
            ret.append([voting_results[0][i],voting_results[1][i],\
                        voting_results[2][i],voting_results[3][i]])
    print("Clustering rate: ",len(ret)," out of ",len(mark))
    return ret

def average_voting(clustering):
    size = len(clustering)
    ave_rot = np.zeros((3,3),np.float32)
    for i in clustering:
        ave_rot += i[0]
    ave_rot /= size
    W, _, Vt = np.linalg.svd(ave_rot)
    num = len(clustering[0][2])
    landmarks = []
    for i in range(num):
        landmarks.append(np.zeros((3,),np.float32))
    for i in range(size):
        for j in range(num):
            landmarks[j] += clustering[i][2][j]
    for j in range(num):
        landmarks[j] /= size
    ret = (np.dot(W,Vt),landmarks)
    return ret

def estimate_pupil(pcd_full,voting_res):
    left, right = voting_res[1][3:5]
    R = voting_res[0]
    vertices = np.asarray(pcd_full.points)
    kdTree = KDTree(vertices)
    neighbors_ind = kdTree.query_ball_point(left,20000)
    neighbors = vertices[neighbors_ind]
    print(neighbors.shape)
    neigh_z = np.zeros((neighbors.shape[0],))
    for i in range(neighbors.shape[0]):
        neigh_z[i] = np.dot(R,neighbors[i].T).T[2]
    print(neigh_z)
    targets = np.argsort(neigh_z)[-5:-1]
    print(targets)
    print(neighbors[targets])

# return 4 arrays: datas from multiple votings
# Orientations, estimated centroids and landmarks(0,1,2,3,4), and sample triangles
# mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
def voting(save_folder,pcd,landmk_index,h,cluster_num = 50,tlr = 5000):
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
    votes = multi_vote(cluster_num,flann,pcd,vectors,triangles,tlr,landmk_index, h)
    filtered_votes = average_voting(cluster(votes))
    # print("Real location: ",vertices[4280])
    print("Voting Time: ",time.time()-t)
    return filtered_votes

# print(vertices[12278])
# p1 = vertices[12278] - vertices[4280] # base
# p2 = vectors[0][5]-vectors[0][4] # corres
# print(p1)
# print(p2)

if __name__ == "__main__":
    plydata = o3d.io.read_triangle_mesh(proto_folder+"27/rnd_head_6.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = plydata.vertices
    pcd1 = pcd.voxel_down_sample(5000)
    res = voting(save_folder,pcd1,5,1,tlr = 5000)
    print(res[0])
    # estimate_pupil(pcd,res)





