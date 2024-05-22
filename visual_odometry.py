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
        self.fast = cv2.FastFeatureDetector_create(threshold=20)

        # Initialize other Parameters
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.n_features = 0
    
    def track_features(self, time):
        # Use FAST detector to find the keypoints in the first image. Reshape to a 2D point vector.
        if self.n_features < 2000:
            kp_a = self.fast.detect(self.image_a, None)
            self.kp_a = np.array([kp.pt for kp in kp_a], dtype=np.float32).reshape(-1, 1, 2)
        # Compute feature tracking using KLT algorithm:
        self.kp_b, status, err = cv2.calcOpticalFlowPyrLK(self.image_a, self.image_b, self.kp_a, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.kp_aa = self.kp_a[status == 1]
        self.kp_bb = self.kp_b[status == 1]
        self.n_features = self.kp_bb.shape[0]

        E, _ = cv2.findEssentialMat(self.kp_bb, self.kp_aa, self.K, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E=E, points1=self.kp_aa, points2=self.kp_bb, cameraMatrix=self.K, mask=None)
        if time == 1:
            self.R = R
            self.t = t
        else:
            scale = self.get_scale(time)
            if (scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + scale*self.R.dot(t)
                self.R = self.R.dot(R)
    
    def get_scale(self, time):
        # Get XYZ coordinates at t-1, t, and then compute the scale.
        prev_pose = self.pose_data[time-1][:3, 3]
        curr_pose = self.pose_data[time][:3, 3]
        scale = np.linalg.norm(curr_pose - prev_pose)
        return scale

def main():
    VO = MonoVO("00")
    trajectory = [VO.t[:3, 2].copy()]
    actual_trajectory = [pose[:3, 3] for pose in VO.pose_data]
    for t in range(1, len(VO.images)):
        print("Timestep: " + str(t) + "/" + str(len(VO.images)))
        VO.image_a = VO.images[t-1]
        VO.image_b = VO.images[t]
        VO.track_features(t)
        trajectory.append(-1 * VO.t[:3,0].copy())

    trajectory = np.array(trajectory)
    actual_trajectory = np.array(actual_trajectory)

    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 2], label='Estimated Trajectory', color='b')
    plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 2], label='Actual Trajectory', color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Estimated vs Actual Trajectory (X-Y Components)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()