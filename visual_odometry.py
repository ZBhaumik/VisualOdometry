import numpy as np
import os
import cv2

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

        # Initialize Feature Detector
        self.fast = cv2.FastFeatureDetector_create(threshold=20)
    
    def track_features(self, t):
        # This method takes the frames at timestep t-1, t, and tracks the features.
        image_a = self.images[t-1]
        image_b = self.images[t]
        # Use FAST detector to find the keypoints in the first image. Reshape to a 2D point vector.
        kp_a = self.fast.detect(image_a, None)
        kp_a = np.array([kp.pt for kp in kp_a], dtype=np.float32).reshape(-1, 1, 2)
        # Compute feature tracking using KLT algorithm:
        kp_b, status, err = cv2.calcOpticalFlowPyrLK(image_a, image_b, kp_a, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), flags=0, minEigThreshold=0.001)
        # Return the (successfully) tracked keypoints in both images.
        return kp_a[status == 1], kp_b[status == 1] 

def main():
    VO = MonoVO("00")
    VO.track_features(1)

if __name__ == "__main__":
    main()