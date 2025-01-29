import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("V_SLAM_Processed/enabled_BA_pcd.pcd")
print(len(pcd.points))
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
print(len(pcd.points))
#pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=10)
#pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-200, -200, -200], [200, 200, 200]))
print(len(pcd.points))
o3d.visualization.draw_geometries([pcd])