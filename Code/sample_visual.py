import open3d as o3d
import numpy as np

def vis_sample(index):
    num,ind = divmod(index,100000)
    num1,_ = divmod(ind,10000)
    mesh = o3d.io.read_triangle_mesh("./data/prototype/rnd_heads/"+str(num+1)+"/"+"rnd_head_"+str(num1)+".ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    with open("./data/library/LibData/Triangles.npy","rb") as f:
        triangles = np.load(f)
    tri = o3d.geometry.LineSet()
    tri.points = o3d.utility.Vector3dVector(triangles[index])
    tri.lines = o3d.utility.Vector2iVector(np.array([[0,1],[0,2],[1,2]]))
    tri.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    o3d.visualization.draw_geometries([pcd,tri])

if __name__ == "__main__":
    vis_sample(1159338)