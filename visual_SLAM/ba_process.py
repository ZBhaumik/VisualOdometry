from desc import Descriptor
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
# Process bundle adjustment after the fact, because I have collected data already (takes ~30 mins).

def filter_pcd(descriptor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([p.pt[:3] for p in descriptor.points]))
    pcd_filtered, index = pcd.remove_radius_outlier(nb_points=6, radius=10)

    index_set = set(index)
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

    for frame in descriptor.frames:
        for pt_idx, point in enumerate(frame.pts):
            if point is not None and point.id in old_to_new_id:
                frame.pts[pt_idx] = descriptor.points[old_to_new_id[point.id]]
            else:
                frame.pts[pt_idx] = None
    return pcd_filtered

descriptor = Descriptor(None)
descriptor.load_pickle("full_data_00.pkl")
pcd = filter_pcd(descriptor)
print(len(descriptor.points))
o3d.visualization.draw_geometries([pcd])
descriptor.pickle("filtered_data_00.pkl")
a, b = descriptor.bundle_adjustment()
plt.plot(a)
plt.show()
plt.plot(b.fun)
plt.show()
descriptor.save_state()