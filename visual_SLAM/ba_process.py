from desc import Descriptor
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import time
# Process bundle adjustment after the fact, because I have collected data already (takes ~30 mins).

def filter_pcd(descriptor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([p.pt[:3] for p in descriptor.points]))
    pcd2, index = pcd.remove_radius_outlier(nb_points=6, radius=10)
    l = []
    idx = 0
    for point in descriptor.points:
        time.sleep(1)
        print(point.id)
        if point.id in index:
            # We are keeping the point.
            l.append(point)
        if point.id not in index:
            print("no")
            i += 1
            point.delete()
    descriptor.points = l
    return pcd2

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