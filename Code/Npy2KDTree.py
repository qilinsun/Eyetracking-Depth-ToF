import numpy as np
import pyflann
import time
folders = 49
main_folder = "./data/library/PositionLib/"
save_folder = "./data/library/LibData/"
m = 10 # Number of prototype files used in each generation
n = 10000 # Number of triangles sampled from each prototype
k = 5
cks = 100

# descriptors = np.zeros((m*n,k**2),np.float32)
# triangles = np.zeros((m*n,3,3),np.float32)
# vectors = np.zeros((m*n,5,3),np.float32)

descriptors = np.zeros((folders*m*n,k**2),np.int32)
triangles = np.zeros((folders*m*n,3,3),np.int32)
vectors = np.zeros((folders*m*n,5,3),np.int32)
t = time.time()
for i in range(folders):
    index = i+1
    with open(main_folder + str(index) + "/" + "Descriptors.npy",'rb') as f:
        descriptors[i*m*n:(i+1)*m*n,:] = np.load(f)[0:m*n,:]
    with open(main_folder + str(index) + "/" + "Triangles.npy",'rb') as f:
        triangles[i*m*n:(i+1)*m*n,:] = np.load(f)[0:m*n,:]
    with open(main_folder + str(index) + "/" + "Vectors.npy",'rb') as f:
        vectors[i*m*n:(i+1)*m*n,:] = np.load(f)[0:m*n,:]
        
flann = pyflann.FLANN()
params = flann.build_index(descriptors, algorithm = 'kdtree', checks = cks)

flann.save_index(save_folder+"Index")
with open(save_folder + "Descriptors.npy", 'wb') as f:
    np.save(f,descriptors)
with open(save_folder + "Vectors.npy", 'wb') as f:
    np.save(f,vectors)
with open(save_folder + "Triangles.npy", 'wb') as f:
    np.save(f,triangles)
