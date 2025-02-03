from desc import Descriptor
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
# Process bundle adjustment after the fact, because I have collected data already (takes ~30 mins).

def filter_pcd(descriptor):
    # Use this filtering method if the pointcloud you collect is somewhat noisy. After making changes
    # to the SLAM system, my PCDs became less noisy, so this isn't as necessary anymore.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([p.pt[:3] for p in descriptor.points]))
    print(len(pcd.points))
    pcd1, i1 = pcd.remove_radius_outlier(nb_points=6, radius=10)
    print(len(pcd1.points))
    pcd2, i2 = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(len(pcd2.points))
    pcd3 = pcd.uniform_down_sample(2)
    i3 = np.arange(len(pcd.points))[::2]
    print(len(pcd3.points))

    index_set = set(np.intersect1d(i1, np.intersect1d(i2, i3)))
    pcd_filtered = pcd.select_by_index(list(index_set))
    new_points = []
    old_to_new_id = {}
    new_id = 0

    for point in descriptor.points:
        if point.id in index_set:
            old_to_new_id[point.id] = new_id  # Store new ID mapping
            point.id = new_id  # Update point ID
            new_points.append(point)
            new_id += 1
        else:
            point.delete()
    descriptor.points = new_points
    return pcd_filtered

"""
    for frame in descriptor.frames:
        for pt_idx, point in enumerate(frame.pts):
            if point is not None and point.id in old_to_new_id:
                frame.pts[pt_idx] = descriptor.points[old_to_new_id[point.id]]
            else:
                frame.pts[pt_idx] = None

"""


descriptor = Descriptor(None)
descriptor.load_pickle("filtered_data_00.pkl")
#pcd = filter_pcd(descriptor)
#o3d.visualization.draw_geometries([pcd])
#descriptor.pickle("filtered_data_00.pkl")
a, b = descriptor.bundle_adjustment()
plt.plot(a)
plt.show()
plt.plot(b.fun)
plt.show()
descriptor.save_state()