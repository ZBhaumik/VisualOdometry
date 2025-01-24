from scipy.optimize import least_squares
from multiprocessing import Process, Queue
import numpy as np
import open3d as o3d

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
        self.q = None
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
        """Create a separate process for 3D viewer."""
        self.q = Queue()
        self.vp = Process(target=self._viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def _viewer_thread(self, q):
        """Thread that updates the 3D viewer."""
        self._viewer_init(1024, 768)
        while True:
            self._viewer_refresh(q)

    def _viewer_init(self, w, h):
        """Initialize the 3D viewer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=w, height=h)
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_poses = []
        self.vis.add_geometry(self.point_cloud)

    def _viewer_refresh(self, q):
        """Refresh the 3D viewer with new data."""
        if self.state is None or not q.empty():
            self.state = q.get()

        poses, points = self.state
        # Update point cloud
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.vis.update_geometry(self.point_cloud)

        # Update camera poses (optional: visualize as coordinate frames)
        for pose in poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            frame.transform(pose)
            self.vis.add_geometry(frame)

        self.vis.poll_events()
        self.vis.update_renderer()

    def display(self):
        """Display the current state in the viewer."""
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))