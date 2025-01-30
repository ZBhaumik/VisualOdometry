import numpy as np
import cv2

def featureMapping(image):
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(image.astype(np.uint8), 1000, qualityLevel=0.01, minDistance=8)
  key_pts = [cv2.KeyPoint(point[0][0], point[0][1], 10) for point in pts]
  key_pts, descriptors = orb.compute(image, key_pts)
  return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

def normalize(K_inv, pts):
  return np.dot(K_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  ret /= ret[2]
  return int(round(ret[0])), int(round(ret[1]))

Identity = np.eye(4)
class Camera(object):
    def __init__(self, desc_dict, image, K):
        self.K = K
        self.pose = Identity
        self.h, self.w = image.shape[0:2]    
        key_pts, self.descriptors = featureMapping(image)
        self.key_pts = normalize(np.linalg.inv(self.K), key_pts)
        self.pts = [None]*len(self.key_pts)
        self.id = len(desc_dict.frames)
        desc_dict.frames.append(self)


"""
Identity = np.eye(4)
class Camera(object):
  def __init__(self, desc_dict, image_t_1, image, K):
      self.K = K
      self.pose = Identity
      self.h, self.w = image.shape[0:2]   
      self.fast = cv2.FastFeatureDetector_create(threshold=25)
      self.features = 0
      key_pts, self.descriptors = self.featureMapping(image_t_1, image)
      self.key_pts = normalize(np.linalg.inv(self.K), key_pts)
      self.pts = [None]*len(self.key_pts)
      self.id = len(desc_dict.frames)
      desc_dict.frames.append(self)
  def featureMapping(self, image_t_1, image):
    # Use FAST detector to find the keypoints in the first image. Reshape to a 2D point vector.
    if self.features < 2000:
        kp_a = self.fast.detect(image_t_1, None)
        self.kp_a = np.array([kp.pt for kp in kp_a], dtype=np.float32).reshape(-1, 1, 2)
    # Compute feature tracking using KLT algorithm:
    self.kp_b, status, err = cv2.calcOpticalFlowPyrLK(image_t_1, image, self.kp_a, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # Return the (successfully) tracked keypoints in both images.
    self.features = self.kp_b[status == 1].shape[0]
    keypoints = [cv2.KeyPoint(pt[0], pt[1], 10) for pt in self.kp_b[status == 1]]
    # Compute descriptors using ORB (or use SIFT/BRISK depending on the use case)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.compute(image, keypoints)
    return self.kp_b[status == 1], descriptors"""