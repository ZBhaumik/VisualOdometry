import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class MonoVO():
    def __init__(self, dataset_id):
        # Load Data
        os.chdir("..")
        DATASET = "//" + dataset_id
        path = os.path.abspath(os.curdir) + DATASET + "//image_0"
        image_paths = os.listdir(path)
        self.images = np.load('VisualSLAM//images_00.npy')#np.array([cv2.imread(path + "//" + image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths])

        # Initialize Calibration
        with open(os.path.abspath(os.curdir) + DATASET + "//calib.txt", 'r') as calib:
            calib_params = [float(item) for item in calib.readline().split(' ')[1:]]
            self.P = np.reshape(calib_params, (3,4))
            self.K = self.P[:3, :3]

        # Initialize Pose Data
        poses = []
        with open(os.path.abspath(os.curdir) + DATASET + "//poses.txt", 'r') as pose_data:
            for line in pose_data:
                pose_param = [float(coord) for coord in line.split(' ')]
                pose_param = np.reshape(pose_param, (3,4))
                #vstack to form a 4x4 pose matrix
                pose_param = np.vstack((pose_param, [0,0,0,1]))
                poses.append(pose_param)
        
        self.pose_data = poses

        # Initialize Feature Detector
        self.fast = cv2.FastFeatureDetector_create(threshold=25)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.features = 0

        self.global_map = np.empty((0, 3))
        self.global_descriptors = None
        self.prev_keypoints = None
        self.prev_descriptors = None
    
    def track_features(self, t):
        # This method takes the frames at timestep t-1, t, and tracks the features.
        image_a = self.images[t-1]
        image_b = self.images[t]
        # Use FAST detector to find the keypoints in the first image. Reshape to a 2D point vector.
        if self.features < 2000:
            kp_a = self.fast.detect(image_a, None)
            self.kp_a = np.array([kp.pt for kp in kp_a], dtype=np.float32).reshape(-1, 1, 2)
        # Compute feature tracking using KLT algorithm:
        self.kp_b, status, err = cv2.calcOpticalFlowPyrLK(image_a, image_b, self.kp_a, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # Return the (successfully) tracked keypoints in both images.
        self.features = self.kp_b[status == 1].shape[0]
        return self.kp_a[status == 1], self.kp_b[status == 1]
    
    def estimate_pose(self, kp_a, kp_b):
        E, _ = cv2.findEssentialMat(kp_b, kp_a, self.K, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E=E, points1=kp_a, points2=kp_b, cameraMatrix=self.K, mask=None)
        return R, np.array([-1 * t[0], t[1], np.abs(t[2])])
    
    def get_scale(self, t):
        # Get XYZ coordinates at t-1, t, and then compute the scale.
        prev_pose = self.pose_data[t-1][:3, 3]
        curr_pose = self.pose_data[t][:3, 3]
        scale = np.linalg.norm(curr_pose - prev_pose)
        return scale
    
    def update_map(self, P_a, P_b, t):
        image_a = self.images[t-1]
        image_b = self.images[t]
        kp_a, des_a = self.orb.detectAndCompute(image_a, None)
        kp_b, des_b = self.orb.detectAndCompute(image_b, None)
        
        matches = self.bf.match(des_a, des_b)
        matches = sorted(matches, key=lambda x: x.distance)
        
        kp_a_m = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        kp_b_m = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        
        self.prev_keypoints = kp_b
        self.prev_descriptors = des_b
        triangulation = cv2.triangulatePoints(P_a, P_b, kp_a_m.T, kp_b_m.T)
        tr_tf = cv2.convertPointsFromHomogeneous(triangulation.T)
        new_pts = tr_tf.squeeze()

        if self.global_descriptors is None:
            self.global_map = new_pts
            self.global_descriptors = self.prev_descriptors
        else:
            self.global_map = np.vstack((self.global_map, new_pts))
            self.global_descriptors = np.vstack((self.global_descriptors, self.prev_descriptors))

def main():

    # Initialize
    VO = MonoVO("00")
    rot = np.zeros(shape=(3,3))
    trans = np.zeros(shape=(3,3))
    trajectory = [trans[:3, 2].copy()]
    actual_trajectory = [pose[:3, 3] for pose in VO.pose_data]

    # Set up visualization
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax_plot = axs[0]
    ax_plot.set_xlabel('X')
    ax_plot.set_ylabel('Y')
    ax_plot.set_title('Estimated vs Actual Trajectory (X-Y Components)')
    ax_plot.grid(True)
    estimated_plot, = ax_plot.plot([], [], label='Estimated Trajectory', color='b')
    actual_plot, = ax_plot.plot([], [], label='Actual Trajectory', color='r')
    ax_plot.legend()
    ax_image = axs[1]
    img_plot = ax_image.imshow(VO.images[0], cmap='gray')
    ax_image.set_title('Current Frame')
    
    for t in range(1, len(VO.images)):
        print("Timestep: " + str(t) + "/" + str(len(VO.images)))
        kp_a, kp_b = VO.track_features(t)
        rotation, translation = VO.estimate_pose(kp_a, kp_b)
        scale = VO.get_scale(t)

        # Compute project matrices, update coordinates, map observations
        P_a = np.eye(4)

        if t == 1:
            rot = rotation
            trans = translation
        else:
            P_a[:3, :3] = rot
            P_a[:3, 3] = trans.flatten()
            if (scale > 0.1 and abs(translation[2][0]) > abs(translation[0][0]) and abs(translation[2][0]) > abs(translation[1][0])):
                trans = trans + scale*rot.dot(translation)
                rot = rot.dot(rotation)
        
        P_b = np.eye(4)
        P_b[:3, :3] = rot
        P_b[:3, 3] = trans.flatten()

        trajectory.append(trans[:3, 0].copy())

        # Update plots with new data
        trajectory_np = np.array(trajectory)
        actual_trajectory_np = np.array(actual_trajectory[:t + 1])
        estimated_plot.set_data(trajectory_np[:, 0], trajectory_np[:, 2])
        actual_plot.set_data(actual_trajectory_np[:, 0], actual_trajectory_np[:, 2])
        img_plot.set_data(VO.images[t])
        ax_plot.relim()
        ax_plot.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    

if __name__ == "__main__":
    main()