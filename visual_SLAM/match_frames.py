import cv2
import numpy as np
from skimage.measure import ransac
from camera import denormalize
from skimage.transform import EssentialMatrixTransform

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(F):
  W = np.asmatrix([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,_,Vt = np.linalg.svd(F)
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret


# My attempt to write a more modern relative pose estimation function, but it turns out it's just the same math.
def relative_pose(f1, f2, K):
  # Perform BF feature matching
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)
  # Relative pose estimation
  ret = []
  x1, x2 = [], []
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      p1 = f1.key_pts[m.queryIdx]
      p2 = f2.key_pts[m.trainIdx]
      if np.linalg.norm((p1-p2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
        if m.queryIdx not in x1 and m.trainIdx not in x2:
          x1.append(m.queryIdx)
          x2.append(m.trainIdx)
          ret.append((p1, p2))
  ret = np.array(ret, dtype=np.float32)
  x1 = np.array(x1)
  x2 = np.array(x2)

  a1 = np.array([denormalize(K, pt1) for pt1 in ret[:, 0]])
  a2 = np.array([denormalize(K, pt2) for pt2 in ret[:, 1]])
  E, _ = cv2.findEssentialMat(a1, a2, K, method=cv2.RANSAC, prob=0.999, threshold=1)
  _, R, t, inlier_mask = cv2.recoverPose(E, a1, a2, K)

  inlier_indices = inlier_mask.ravel().astype(bool)
  
  transformation = np.eye(4)
  transformation[:3, :3] = R
  transformation[:3, 3] = t.ravel()
  
  return x1[inlier_indices], x2[inlier_indices], transformation

def generate_match(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)
  ret = []
  x1,x2 = [], []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      pts1 = f1.key_pts[m.queryIdx]
      pts2 = f2.key_pts[m.trainIdx]
      if np.linalg.norm((pts1-pts2)) < 0.1*np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
        if m.queryIdx not in x1 and m.trainIdx not in x2:
          x1.append(m.queryIdx)
          x2.append(m.trainIdx)
          ret.append((pts1, pts2))
  assert(len(set(x1)) == len(x1))
  assert(len(set(x2)) == len(x2))
  assert len(ret) >= 8
  ret = np.array(ret)
  x1 = np.array(x1)
  x2 = np.array(x2)
  model, f_pts = ransac((ret[:, 0], ret[:, 1]), EssentialMatrixTransform, min_samples=8, residual_threshold=0.001, max_trials=100)
  Rt = extractRt(model.params)
  return x1[f_pts], x2[f_pts], Rt