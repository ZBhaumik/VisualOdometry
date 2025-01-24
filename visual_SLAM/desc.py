from multiprocessing import Process, Queue
import numpy as np
import open3d as o3d

class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

  def __init__(self, mapp, loc):
    self.pt = loc
    self.frames = []
    self.idxs = []
    
    self.id = len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

class Descriptor(object):
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = None
  # G2O optimization:
  def optimize(self):
    err = optimize(self.frames, self.points, local_window, fix_points, verbose, rounds)

    # Key-Point Pruning:
    culled_pt_count = 0
    for p in self.points:
      # <= 4 match point that's old
      old_point = len(p.frames) <= 4 and p.frames[-1].id+7 < self.max_frame
      #handling the reprojection error
      errs = []
      for f,idx in zip(p.frames, p.idxs):
        uv = f.kps[idx]
        proj = np.dot(f.pose[:3], p.homogeneous())
        proj = proj[0:2] / proj[2]
        errs.append(np.linalg.norm(proj-uv))
      if old_point or np.mean(errs) > CULLING_ERR_THRES:
        culled_pt_count += 1
        self.points.remove(p)
        p.delete()

    return err

  def create_viewer(self):
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window(window_name="3D Viewer", width=w, height=h)
    self.point_cloud = o3d.geometry.PointCloud()
    self.camera_poses = []
    self.vis.add_geometry(self.point_cloud)

  def viewer_refresh(self, q):
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
    if self.q is None:
      return
    poses, pts = [], []
    for f in self.frames:
      poses.append(f.pose)
    for p in self.points:
      pts.append(p.pt)
    self.q.put((np.array(poses), np.array(pts)))