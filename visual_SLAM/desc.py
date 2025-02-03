from scipy.optimize import least_squares
import numpy as np
import open3d as o3d
import time
import cv2
from camera import denormalize
from scipy.sparse import lil_matrix
import pickle

class Point:
    def __init__(self, m1, l1):
        self.pt = l1
        self.id = len(m1.points)
        m1.points.append(self)
        self.frames = []
        self.idxs = []

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

    def homogeneous(self):
        return np.array([self.pt[0], self.pt[1], self.pt[2], 1])

    def delete(self):
        """Delete this point's observations and remove it from the map."""
        for idx, frame in zip(self.idxs, self.frames):
            frame.pts[idx] = None
        self.frames.clear()
        self.idxs.clear()
    
    def __getstate__(self):
        list_a = [frame.id for frame in self.frames]
        self.frames = list_a
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

class Descriptor:
    def __init__(self, K):
        self.frames = []
        self.points = []
        self.state = None
        self.point_cloud = None
        self.K = K
        self.old_points = []
        self.old_frames = []

    def rotate(self, points, rot_vecs):
        # Rodrigues rotation formula from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    
    def project(self, points, camera_params):
        """Convert 3-D points (world coordinates) to 2-D image coordinates."""
        R_vecs = camera_params[:, :3]
        t_vecs = camera_params[:, 3:6]
        points_proj = np.zeros((points.shape[0], 2))
        
        for i in range(len(camera_params)):
            pose = np.eye(4)
            pose[:3, :3], _ = cv2.Rodrigues(R_vecs[i])
            pose[:3, 3] = t_vecs[i]
            pose_inv = np.linalg.inv(pose)
            points_cam = np.dot(pose_inv[:3, :3], points[i].T).T + pose_inv[:3, 3]
            points_img = (self.K @ points_cam.T).T
            points_img = points_img[:2] / points_img[2, np.newaxis]
            points_proj[i] = points_img
        
        
        #pose = np.eye(4)
        #pose[:3, :3] = cv2.Rodrigues(camera_params[:, :3])
        #pose[:3, 3] = camera_params[:, 3:6]
        #pose_inv = np.linalg.inv(pose)
        #points_proj = np.dot(pose_inv[:3, :3], points.T).T + pose_inv[:3, 3]
        """
        points_proj = self.rotate(points, camera_params[:, :3]) # Rotate
        points_proj += camera_params[:, 3:6] # Translate
        points_proj = np.dot(points_proj, self.K.T) # Project
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        """
        return points_proj
    
    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()
    
    def sparse_jacobian(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        return A
    

    def bundle_adjustment(self):
        # Construct relevant variables.
        points_3d = np.array([p.pt[:3] for p in self.points])

        n_cameras = len(self.frames)
        camera_params = np.zeros((n_cameras, 6))
        for i, frame in enumerate(self.frames):
            # Extract rotation matrix and convert to Rodrigues vector
            rotation_matrix = frame.pose[:3, :3]
            rodrigues_vector, _ = cv2.Rodrigues(rotation_matrix)
            translation_vector = frame.pose[:3, 3]
            camera_params[i, :3] = rodrigues_vector.flatten()
            camera_params[i, 3:6] = translation_vector
        camera_ind = []
        point_ind = []
        points_2d = []
        for frame_idx, frame in enumerate(self.frames):
            for pt_idx, point in enumerate(frame.pts):
                if point is not None:
                    camera_ind.append(frame_idx)
                    point_ind.append(point.id)
                    points_2d.append(denormalize(self.K, frame.key_pts[pt_idx]))

        camera_ind = np.array(camera_ind)
        point_ind = np.array(point_ind)
        print(len(point_ind))
        points_2d = np.array(points_2d)

        n = 6 * n_cameras + 3 * points_3d.shape[0]
        m = 2 * points_2d.shape[0]
        print("n_cameras: {}".format(n_cameras))
        print("n_points: {}".format(points_3d.shape[0]))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        print(points_2d)
        print(self.project(points_3d[point_ind], camera_params[camera_ind]))
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, points_3d.shape[0], camera_ind, point_ind, points_2d)
        t0 = time.time()
        A = self.sparse_jacobian(n_cameras, points_3d.shape[0], camera_ind, point_ind)
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, points_3d.shape[0], camera_ind, point_ind, points_2d))
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        # Extract optimized parameters
        optimized_camera_params = res.x[:6 * n_cameras].reshape((n_cameras, 6))
        optimized_points_3d = res.x[6 * n_cameras:].reshape((-1, 3))

        # Update camera parameters
        for i, frame in enumerate(self.frames):
            rodrigues_vector = optimized_camera_params[i, :3]
            rotation_matrix, _ = cv2.Rodrigues(rodrigues_vector)
            translation_vector = optimized_camera_params[i, 3:6]
            frame.pose[:3, :3] = rotation_matrix
            frame.pose[:3, 3] = translation_vector

        # Update 3D points
        for i, point in enumerate(self.points):
            point.pt[:3] = optimized_points_3d[i]

        print("Optimization complete. Parameters updated.")
        return f0, res
    
    def create_viewer(self):
        """Initialize the 3D viewer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=1024, height=768)
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        self.vis.run()
    
    def update(self):
        self.state = (np.array([p.pt for p in self.points]), np.array([frame.pose for frame in self.frames]))
        if self.state is None:
            return
        points, poses = self.state
        self.point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        return poses

    def update_viewer(self):
        """Update the 3D viewer with new data."""
        poses = self.update()
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame.transform(poses[-1])
        self.vis.add_geometry(frame)

        self.vis.update_geometry(self.point_cloud)  
        self.vis.poll_events()

        self.vis.update_renderer()
        self.vis.run()

    def pickle(self, filename="full_data_00.pkl"):
        data = {
            "frames": self.frames,
            "points": self.points,
            "K": self.K
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Descriptor state saved to {filename}")
    
    def load_pickle(self, filename="full_data_00.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.K = data["K"]  
        self.frames = data["frames"]
        self.points = data["points"]
        frame_dict = {frame.id: frame for frame in self.frames}
        for point in self.points:
            list_a = [frame_dict[frame_id] for frame_id in point.frames if frame_id in frame_dict]
            point.frames = list_a
    
    def save_state(self):
        # Save a lot of data to analyse later because this takes a LONG time to run.
        new_pt_cld = o3d.geometry.PointCloud()
        new_pt_cld.points = o3d.utility.Vector3dVector(np.array([p.pt[:3] for p in self.points]))
        o3d.io.write_point_cloud("saved_pcd.pcd", new_pt_cld)
        np.save("saved_pts.npy", new_pt_cld.points)
        np.save("saved_poses.npy", np.array([frame.pose for frame in self.frames]))  