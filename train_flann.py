import numpy as np
import open3d as o3d
from random import randint
import math
import icp
from sklearn.neighbors import KDTree, NearestNeighbors
import pickle
import os
import time
import pyflann

## Parameter Setting ##
# Parameter of the camera:
# Intrinsic, set to Kinect2 Depth Camera parameters by default

# Path of the images
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head_1.ply"
save_folder = "./data/library/CurrentLib/"


# Get images
# depth_img: np array
# depth_img = io.imread(input_folder+depth_img_name)
def readPLY(face_name):
    plydata = o3d.io.read_triangle_mesh(proto_folder+face_name)
    plydata.compute_triangle_normals()
    return plydata



def getMeshSize(mesh):
    return np.asarray(mesh.triangles).shape[0]

def getVertexSize(mesh):
    return np.asarray(mesh.vertices).shape[0]

# Given the three vertices of a triangle mesh,
# return a random point in the triangle
# Parameters:
#    vertex: an ndarray of length 3 containing the indices of the vertices
#    mesh: the mesh of the face
def getSampleMesh(mesh,vertex):
    point1 = mesh.vertices[vertex[0]][0:3]
    point2 = mesh.vertices[vertex[1]][0:3]
    point3 = mesh.vertices[vertex[2]][0:3]
    alpha = randint(0,100)/100
    beta = randint(0,100-int(100*alpha))/100
    ret = alpha*point1 + beta*point2 + (1-alpha-beta)*point3
    return ret

def rotation(v,n,theta):
    theta = theta * math.pi/180
    ret = v*np.cos(theta) + np.cross(n,v)*np.sin(theta) + \
        np.dot(n,v)*n*(1-np.cos(theta))
    return ret

# l: the side length of the equilateral triangle sample
def sampleFromMesh(mesh,l):
    index = randint(0,getMeshSize(mesh)-1)
    vertex = mesh.triangles[index]
    point = getSampleMesh(mesh,vertex)
    normal = mesh.triangle_normals[index] # normalized normal vector
    r = np.random.random(3)
    r_parallel = r - np.dot(normal,r)*normal
    v1 = r_parallel/np.sqrt(np.dot(r_parallel,r_parallel)) * l / 1.732
    v2 = rotation(v1,normal,120)
    v3 = rotation(v1,normal,240)
    eqTriangle = np.array((point+v1,point+v2,point+v3))
    return eqTriangle

def ICPtransform(pointCloud,triangle):
    flann = pyflann.FLANN()
    # neigh = NearestNeighbors(n_neighbors=1)
    # neigh.fit(pointCloud)
    # a, indices = neigh.kneighbors(triangle, return_distance=True)
    # print("original distances: ",a,indices)
    indices,distances = flann.nn(
    pointCloud, triangle, 1, algorithm="kmeans", branching=32, iterations=7, checks=16)
    distances = np.sqrt(distances)
    # print("original: ",distances)
    target = np.zeros_like(triangle)
    target_set = indices.ravel()
    target[0] = pointCloud[target_set[0]]
    target[1] = pointCloud[target_set[1]]
    target[2] = pointCloud[target_set[2]]
    trans,_,_ = icp.icp(triangle,target,max_iterations=3)
    # temp = np.ones((4,triangle.shape[0]))
    temp = np.ones((4,4))
    temp[:3,:3] = triangle.T
    temp[3,3] = 1
    ret = np.dot(trans,temp)
    ret = ret[:3,:3].T
    _,distances = flann.nn(
    pointCloud, ret, 1, algorithm="kmeans", branching=32, iterations=7, checks=16)
    distances = np.sqrt(distances)
    # print("After: ",distances)
    return ret, distances

# Calculate descripter
# Two vertices of and the normal the triangle
def getSurface(p1,p2,normal):
    n = np.cross((p2-p1),normal)
    n = n/np.sqrt(np.dot(n,n))
    constant = 0 - np.dot(n,p1)
    return (n,constant)

def inSurface(point,center,p1,p2,normal):
    n,constant = getSurface(p1,p2,normal)
    if ((np.dot(center,n)+constant)*(np.dot(point,n)+constant) >= 0):
        return True
    return False

def getDistance(p1,p2):
    return np.sqrt(np.sum(np.square(p1-p2)))

def inTriangle(point,center,a,b,c,normal):
    dis = np.dot((point-center),normal)*normal
    proj = point - dis
    n1 = np.cross(proj-a,(b-a))
    n2 = np.cross(c-a,proj-a)
    n3 = np.cross(proj-c,a-c)
    return (np.dot(n1,n2) >= 0) and (np.dot(n1,n3) >= 0) and \
    (np.dot(n2,n3) >= 0)

def getDesIndices(pointCloud,points,tri,l,k):
    # ret = []
    distances = []
    projections = []
    a = tri[0]
    b = tri[1]
    c = tri[2]
    center = (a+b+c)/3
    normal = np.cross((center-a),(center-b))
    normal /= np.sqrt(np.sum(np.square(normal)))
    v1 = c-b
    v2 = b-a
    n1 = np.cross(v1,normal)
    n1 /= np.sqrt(np.sum(np.square(n1)))
    n2 = np.cross(v2,normal)
    n2 /= np.sqrt(np.sum(np.square(n2)))
    pc = np.asarray(pointCloud)
    for i in list(points[0]):
        point = pc[i]
            # ret.append(i)
        x1 = (np.dot(point,n1)-np.dot(a,n1))/(0.866*l/k)
        # x2 = k-int((np.dot(point,n2)-np.dot(c,n2))/(0.866*l/k))-1
        x2 = (np.dot(point,n2)-np.dot(c,n2))/(0.866*l/k)
        x3 = (np.dot(point,n1)-np.dot(a,n1))/(0.866*l/k)
        if (x1 >= 0 and x1 < k and x2 >= 0 and x2 < k and x3 >= 0 and x3 < k):
            x1 = int(x1)
            x2 = k-1-int(x2)
            distances.append(np.dot(point,normal)-np.dot(center,normal))
            projections.append((x1,x2))
    return distances,projections

# k: k^2 represents number of subtriangles in each triangle sample 
def getDescriptor(pointCloud,tri,l,k):
    tree = KDTree(pointCloud,leaf_size = 4)
    center = (tri[0]+tri[1]+tri[2])/3
    index_around = tree.query_radius(center.reshape(1,-1),r = l/1.732)
    desDistances, projs= getDesIndices(pointCloud,index_around,tri,l,k)
    d = dict()
    for i in range(len(projs)):
        d[projs[i]] = []
    for i in range(len(projs)):
        if (d[projs[i]] == []):
            d[projs[i]] = [int(desDistances[i]),1]
        else:
            d[projs[i]][0] += int(desDistances[i])
            d[projs[i]][1] += 1
    for (key,value) in d.items():
        d[key] = value[0]/value[1]
    ret = np.zeros((k,k),np.float32)
    for key in d.keys():
        ret[key[0],key[1]] = d[key]
    padded_ret = np.pad(ret,(1,1),'reflect')
    n = 0
    while ((0 in ret) and n < 10):
        for i in range(k):
            for j in range(k):
                if ret[i,j] == 0:
                    padded_ret[i+1,j+1] = 0.125 *(padded_ret[i,j] + \
                                    padded_ret[i,j+1] + \
                                    padded_ret[i,j+2] + \
                                    padded_ret[i+1,j] + \
                                    padded_ret[i+1,j+2] + \
                                    padded_ret[i+2,j] + \
                                    padded_ret[i+2,j+1] + \
                                    padded_ret[i+2,j+2])
        n += 1
        ret = padded_ret[1:-1,1:-1]
    return ret.flatten()


# print(getDescriptor(ply.vertices,tri,l,k)

# N: deepest point of nasal bridge 8275
# Prn: 鼻尖 8320
# Bilteral:
# p 猜测是pupil(?) 4280 12278
# ps 眼眶上侧 4404 12144
# pi 眼眶下侧 4158 12414
# ex 眼眶外侧 2088 14472
# en 眼眶内侧 5959 10603
# 坐标距离：实际距离 = 1000:1mm

def getDataVector(mesh,tri,l,k):
    pointCloud = mesh.vertices
    cent = np.sum(pointCloud,axis = 0)/np.asarray(pointCloud).shape[0]
    tri_cent = (np.sum(tri,axis=0)/3)
    v_T = getDescriptor(pointCloud,tri,l,k)
    v_c = cent-tri_cent
    v_1 = pointCloud[8275]-tri_cent
    v_2 = pointCloud[8320]-tri_cent
    u_1 = pointCloud[4280]-tri_cent
    u_2 = pointCloud[12278]-tri_cent
    vector = np.array((v_c,v_1,v_2,u_1,u_2))
    return v_T,tri,vector

# The result for a single mesh
def getDataPiece(faceName,n,l,k,d = 3000):
    ret = np.zeros((n,5,3),np.float32)
    triangles = np.zeros((n,3,3),np.float32)
    descriptors = np.zeros((n,k**2),np.float32)
    ply = readPLY(faceName)
    t = time.time()
    for i in range(n):
        tri,distances = ICPtransform(np.asarray(ply.vertices),sampleFromMesh(ply,l))
        while (distances[0]>d or distances[1]>d or distances[2]>d):
            tri,distances = ICPtransform(np.asarray(ply.vertices),sampleFromMesh(ply,l))
        datas = getDataVector(ply,tri,l,k)
        descriptors[i,:],triangles[i],ret[i] = datas
    print(time.time()-t)
    t = time.time()

    return descriptors,triangles,ret

# files = os.listdir(proto_folder)
# n = 1
# for i in files[0:10]:
#     datas = main(i)
#     des_name = './data/library/des_' + str(n) + '.npy' 
#     vec_name = './data/library/vec_' + str(n) + '.npy' 
#     n += 1
#     np.save(des_name,datas[0])
#     np.save(vec_name,datas[1])

# n: number of triangles on one mesh
# l: the side length of the triangle
# k: the number of subtriangles in one triangle
# des: the descriptor of each base triangle
# vec: tri,v_c,v_1,v_2,u_1,u_2
# namely. base triangle, and the vectors starting from center of the tri to:
# center of head, deepest point of nasal bridge, nose tip, left pupil, right pupil
# The indices of des and vec are corresponding
def generateLib(files,n, l, k):
    m = len(files)
    descriptors = np.zeros((m*n,k**2),np.float32)
    triangles = np.zeros((m*n,3,3),np.float32)
    vectors = np.zeros((m*n,5,3),np.float32)
    t = time.time()
    for i in range(m):
        des,tri,vec = getDataPiece(files[i],n,l,k)
        descriptors[i*n:(i+1)*n,:] = des
        triangles[i*n:(i+1)*n] = tri
        # a = tri[0]
        # print(np.sqrt(np.sum(np.square(a[1]-a[0]))),"???")
        vectors[i*n:(i+1)*n] = vec
    # KDTree_des = KDTree(descriptors, leaf_size = 4)
    return descriptors,triangles,vectors

if __name__ == "__main__":
    files = os.listdir(proto_folder)[1:2]
    data = generateLib(files,n = 1000, l = 80000, k = 5)
    # a = data[1][1]
    # print(np.sqrt(np.sum(np.square(a[1]-a[0]))),"???")
    with open(save_folder+'Descriptors_flann.npy', 'wb') as f:
        np.save(f,data[0])
    with open(save_folder+'Vectors.npy', 'wb') as f:
        np.save(f,data[2])
    with open(save_folder+'Triangles.npy', 'wb') as f:
        np.save(f,data[1])
# Average time per mesh: 360s*n/1000 = 0.36n

    
