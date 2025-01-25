from scipy.optimize import least_squares
import numpy as np
import open3d as o3d
import time

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
        for frame in self.frames:
            frame.pts[self.idxs[self.frames.index(frame)]] = None
        self.frames.clear()
        self.idxs.clear()

class Descriptor:
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.point_cloud = None
        self.max_frame = 0  # Set max_frame for point pruning

    def optimize(self, local_window, fix_points, verbose, rounds, culling_threshold):
        """Perform Bundle Adjustment optimization and keypoint pruning."""
        # Set up initial parameters (camera poses and 3D points)
        params = self._setup_initial_params()
        # Perform Bundle Adjustment optimization
        result = self._bundle_adjustment(params, local_window, fix_points, verbose, rounds)
        # Update points and frames with optimized parameters
        self._update_optimized_params(result)
        # Key-Point Pruning
        _ = self._prune_points(culling_threshold)
        
        return result.fun  # Return the final residuals

    def _setup_initial_params(self):
        params = []
        # Add camera poses
        for frame in self.frames:
            params.extend(frame.pose[:3].flatten())
            params.extend(frame.pose[3:].flatten())

        for point in self.points:
            params.extend(point.pt)

        return np.array(params)

    def _bundle_adjustment(self, params, local_window, fix_points, verbose, rounds):
        # Define the cost function
        def cost_function(params):
            residuals = []
            # Update the camera poses and 3D points from the parameters
            idx = 0
            for frame in self.frames:
                frame.pose[:3] = params[idx:idx+3]
                frame.pose[3:] = params[idx+3:idx+6]
                idx += 6

            for point in self.points:
                point.pt = params[idx:idx+3]
                idx += 3

            # Calculate residuals (reprojection errors)
            for p in self.points:
                for f, idx in zip(p.frames, p.idxs):
                    uv = f.kps[idx]  # observed keypoint
                    proj = np.dot(f.pose[:3], p.homogeneous())  # project 3D point to 2D
                    proj = proj[:2] / proj[2]  # Convert to homogeneous coordinates
                    residuals.append(np.linalg.norm(proj - uv))  # Compute the residual (error)

            return np.array(residuals)

        result = least_squares(cost_function, params, verbose=verbose, max_nfev=rounds)

        return result

    def _update_optimized_params(self, result):
        idx = 0
        for frame in self.frames:
            frame.pose[:3] = result.x[idx:idx+3]
            frame.pose[3:] = result.x[idx+3:idx+6]
            idx += 6

        for point in self.points:
            point.pt = result.x[idx:idx+3]
            idx += 3

    def _prune_points(self, culling_threshold):
        """Prune points based on reprojection error and observation age."""
        culled_pt_count = 0
        for p in self.points:
            old_point = len(p.frames) <= 4 and p.frames[-1].id + 7 < self.max_frame
            errs = self._calculate_reprojection_error(p)

            if old_point or np.mean(errs) > culling_threshold:
                culled_pt_count += 1
                self.points.remove(p)
                p.delete()

        return culled_pt_count

    def _calculate_reprojection_error(self, point):
        """Calculate reprojection error for a point across all frames."""
        errs = []
        for f, idx in zip(point.frames, point.idxs):
            uv = f.kps[idx]
            proj = np.dot(f.pose[:3], point.homogeneous())
            proj = proj[:2] / proj[2]
            errs.append(np.linalg.norm(proj - uv))
        return errs
    
    def create_viewer(self):
        """Initialize the 3D viewer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=1024, height=768)
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)
        self.view_control = self.vis.get_view_control()
        self.vis.run()

    def update_viewer(self):
        """Update the 3D viewer with new data."""
        self.state = (np.array([p.pt for p in self.points]), np.array([frame.pose for frame in self.frames]))
        if self.state is None:
            return
        points, poses = self.state
        self.point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

        """
        view_control.set_lookat(poses[-1][:3, 3])
        view_control.set_up(pose[:3, 1])
        view_control.set_front(pose[:3, 2])
        view_control.set_zoom(10)  # Optional: adjust the zoom level
        view_control.set_constant_z_far(200)  # Optional: adjust the far plane
        """

        self.vis.update_geometry(self.point_cloud)
        R = poses[-1][:3, :3]
        R_trans = np.dot([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], R)
        poses[-1][:3, :3] = R_trans
        poses[-1][2, 3] = poses[-1][2, 3] * -1
        poses[-1][0, 3] = poses[-1][0, 3] * -1
        cam = self.view_control.convert_to_pinhole_camera_parameters()
        cam.extrinsic = poses[-1] # where T is your matrix
        self.view_control.convert_from_pinhole_camera_parameters(cam)
        self.view_control.set_constant_z_far(1000)  # Optional: adjust the far plane
        self.view_control.set_zoom(10)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()