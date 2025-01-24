import numpy as np
import cv2

def featureMapping(image):
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(image.astype(np.uint8), 1200, qualityLevel=0.01, minDistance=8)
  key_pts = [cv2.KeyPoint(point[0][0], point[0][1], 10) for point in pts]
  key_pts, descriptors = orb.compute(image, key_pts)
  return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

def normalize(K_inv, pts):
  return np.dot(K_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

def denormalize(count, pt):
  ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
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