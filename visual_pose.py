from getEstimation_flann_abs import voting
import open3d as o3d
import numpy as np

input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/PositionLib/data/"

# Orientation, centroids, 
mesh = o3d.io.read_triangle_mesh(input_folder+test_face)
res = voting(save_folder,mesh,3,1)
vertices = np.array(mesh.vertices)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
o3d.visualization.draw_geometries([pcd])


        

