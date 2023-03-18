from getEstimation_flann_abs import voting
import open3d as o3d
import numpy as np

input_folder = "./data/input/"
proto_folder = "./data/prototype/rnd_heads/"
test_face = "rnd_head.ply"
save_folder = "./data/library/PositionLib/data/"


mesh = o3d.io.read_triangle_mesh(input_folder+test_face)

# Orientations, estimated centroids and landmarks, and sample triangles
res = voting(save_folder,mesh,5,1)
print(res[2][0])
_,nose,nose_tip,right_eye,left_eye = res[2][0]
R = res[0][0]
print(R)
orie_vec = np.dot(R,np.array([0,0,1]))
orie_vec *= 100000
# orie_vec = np.array([[0,orie_vec[0]],[0,orie_vec[1]],[0,orie_vec[2]]])
vertices = np.array(mesh.vertices)

print("Start visualizing...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
# lines = o3d.geometry.LineSet()
lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(np.array([nose,nose_tip,left_eye,right_eye,\
                                                    nose+orie_vec/2,nose_tip+orie_vec,left_eye+orie_vec/2,right_eye+orie_vec/2]))
# lines.paint_uniform_color([1, 0.5, 0])
# mes = o3d.geometry.TriangleMesh()
# mes.vertices = pcd.points
# mes.lines = lines.lines
lines.lines = o3d.utility.Vector2iVector(np.array([[0,4],[1,5],[2,6],[3,7]]))
lines.colors = o3d.utility.Vector3dVector(np.array([[1,0.5,0],[0,0,1],[0,0,0],[0,0,0]]))
o3d.visualization.draw_geometries([lines,pcd])



        

