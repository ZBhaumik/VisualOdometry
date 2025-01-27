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

    def optimize(self, window_size=5):
        """
        Perform local bundle adjustment over a sliding window of frames.
        
        Args:
            window_size (int): Number of recent frames to include in the optimization.
        """
        if len(self.frames) < window_size:
            print("Not enough frames for local optimization.")
            return

        # Select the most recent frames and points observed in those frames
        recent_frames = self.frames[-window_size:]
        recent_frame_ids = [frame.id for frame in recent_frames]
        
        # Find points observed in the selected frames
        recent_points = []
        for point in self.points:
            if any(frame.id in recent_frame_ids for frame in point.frames):
                recent_points.append(point)

        # Prepare data for optimization
        points_3d = np.array([p.pt[:3] for p in recent_points])
        poses = np.array([frame.pose for frame in recent_frames])

        observations = []
        for point in recent_points:
            for frame, idx in zip(point.frames, point.idxs):
                if frame.id in recent_frame_ids:
                    observations.append((recent_points.index(point), recent_frame_ids.index(frame.id), frame.key_pts[idx]))

        def reprojection_error(params):
            n_points = len(recent_points)
            n_frames = len(recent_frames)

            # Extract points and poses from flattened parameters
            points = params[:n_points * 3].reshape((n_points, 3))
            poses = params[n_points * 3:].reshape((n_frames, 4, 4))

            errors = []
            for point_id, frame_id, observed_pt in observations:
                # Project the 3D point into the frame's image plane
                point_h = np.append(points[point_id], 1)  # Homogeneous coordinates
                projected = poses[frame_id] @ point_h
                projected /= projected[2]  # Normalize by depth
                projected_2d = projected[:2]

                error = observed_pt - projected_2d
                errors.extend(error)

            return np.array(errors)

        # Flatten the initial parameters for optimization
        initial_params = np.hstack([points_3d.flatten(), poses.flatten()])

        # Perform bundle adjustment
        result = least_squares(
            reprojection_error,
            initial_params,
            verbose=2
        )

        # Update points and poses with optimized values
        optimized_params = result.x
        n_points = len(recent_points)
        points_optimized = optimized_params[:n_points * 3].reshape((n_points, 3))
        poses_optimized = optimized_params[n_points * 3:].reshape((len(recent_frames), 4, 4))

        for i, point in enumerate(recent_points):
            point.pt = points_optimized[i]

        for i, frame in enumerate(recent_frames):
            frame.pose = poses_optimized[i]

        print(f"Local bundle adjustment completed with cost: {result.cost}")

    
    def create_viewer(self):
        """Initialize the 3D viewer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=1024, height=768)
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        self.vis.run()

    def update_viewer(self):
        """Update the 3D viewer with new data."""
        self.state = (np.array([p.pt for p in self.points]), np.array([frame.pose for frame in self.frames]))
        if self.state is None:
            return
        points, poses = self.state
        self.point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame.transform(poses[-1])
        self.vis.add_geometry(frame)

        self.vis.update_geometry(self.point_cloud)  
        self.vis.poll_events()

        #camera_params = self.view_control.convert_to_pinhole_camera_parameters()
        #extmat = camera_params.extrinsic.copy()
        #extmat[:3, :3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #camera_params.extrinsic = extmat
        #self.view_control.convert_from_pinhole_camera_parameters(camera_params)

        self.vis.update_renderer()
        self.vis.run()
    
    def save_point_cloud(self):
        o3d.io.write_point_cloud("no_bundle_adjustment.pcd", self.point_cloud)
        np.save("no_bundle_adjustment.npy", self.point_cloud.points)
        print(len(self.point_cloud.points))   