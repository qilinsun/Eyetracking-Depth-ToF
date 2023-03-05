import numpy as np
import open3d as o3d
from train_flann import ICPtransform, getDescriptor
from random import randint
from sklearn.neighbors import KDTree
import math
import pyflann
import time


# Path of the images
input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head_1.ply"
save_folder = "./data/library/"
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
    downpcd = o3d.geometry.voxel_down_sample(pcd,voxel_size=0.01)
    # o3d.visualization.draw_geometries([downpcd])
    o3d.geometry.estimate_normals(downpcd,\
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30000, max_nn=30))
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


def getNearDes(des,des_lib):
    # s = pickle.dumps(des_lib)
    # tree_copy = pickle.loads(s)
    # dis,ind = tree_copy.query(des.reshape(1,-1),1)
    flann = pyflann.FLANN()
    testset = np.array([des])
    ind,dis = flann.nn(des_lib,testset,1,algorithm="kmeans",branching = 32, iterations = 7, checks = 16)
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

def vote(vectors,triangles,des,descriptors,tri,landmk_index = -1):
    index,dis= getNearDes(des,descriptors)
    print(index,dis)
    index = index[0]
    corres_vec = vectors[index]
    corres_tri = triangles[index]
    base_cen = (tri[0]+tri[1]+tri[2])/3
    base_cen_tri = tri - base_cen
    corres_cen = (corres_tri[0] + corres_tri[1] + corres_tri[2]) /3
    corres_cen_tri = corres_tri - corres_cen
    # print("tri: ",tri)
    # print("corres: ",corres_tri)
    # print("Corres_Des: ",getDescriptor(vertices,corres_tri,80000,5))
    # trans = np.dot(base_cen_tri.T,np.linalg.inv(corres_cen_tri.T))
    # left_pupil = np.dot(trans,corres_cen_tri.T).T
    # print(left_pupil)
    a = np.array(base_cen_tri,np.int64)
    b = np.array(corres_cen_tri,np.int64)
    # print("tricen: ",a)
    # print("correscen: ",b)
    trans = np.dot(a.T,np.linalg.inv(b.T))
    print("Orientation: ",trans)
    left_pupil = np.dot(trans,np.array(corres_vec[3],np.int64).T).T + base_cen
    print("Estimated Location: ",left_pupil)
    if landmk_index != -1:
        landmark = np.dot(trans,np.array(corres_vec[landmk_index],np.int64).T).T \
            + base_cen
    else:
        landmark = 0
    centroid = np.dot(trans,np.array(corres_vec[0],np.int64).T).T \
        + base_cen
    return trans,centroid,landmark
    
def multi_vote(n,descriptors,vertices,vectors,triangles,landmk_index=-1,l=80000, d=3000, k=5):
    orients = []
    centroids = []
    landmarks = []
    i = 0
    while (i < n):
        try:
            # tri, distances = ICPtransform(\
            #         vertices,sampleFromVertices(vertices,l))
            # tri = triangles[100]
            # distances = [d+1,d+1,d+1]
            # while (distances[0]>d or distances[1]>d or distances[2]>d):
            #     tri,distances = ICPtransform(\
            #         vertices,sampleFromVertices(vertices,l))
            # print("distances: ",distances)
            index = randint(1,200)
            tri = triangles[index]+1000*np.random.random((1,3))
            des = getDescriptor(vertices,tri, l, k)
            # print("Des: ",des)
            ret = vote(vectors,triangles,des,descriptors,tri,landmk_index)
            orients.append(ret[0])
            centroids.append(ret[1])
            landmarks.append(ret[2])
            i += 1
        except:
            continue
    if landmk_index == -1:
        return orients,centroids,[]
    return orients,centroids,landmarks


def isNear(R1, R2, theta):
    fdis = np.linalg.norm(R1-R2)
    print(fdis)
    return (fdis < 2.828*abs(math.sin(theta/360*math.pi)))

with open(save_folder+'Descriptors_flann.npy', 'rb') as f:
    descriptors = np.load(f) # 目前只存了5000个triangle...
with open(save_folder+'Vectors.npy', 'rb') as f:
    vectors = np.load(f) # vec: v_c,v_1,v_2,u_1,u_2
with open(save_folder+'Triangles.npy', 'rb') as f:
    triangles = np.load(f)

t = time.time()
mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
vertices = np.asarray(mesh.vertices)

# orients,centroids,landmarks

votes = multi_vote(1,descriptors,vertices,vectors,triangles,0)

print("Real location: ",vertices[4280])
print("Time used: ",time.time()-t)
# print(vertices[12278])
# p1 = vertices[12278] - vertices[4280] # base
# p2 = vectors[0][5]-vectors[0][4] # corres
# print(p1)
# print(p2)





