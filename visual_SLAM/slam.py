import numpy as np
import cv2
import os
import sys
from camera import denormalize, normalize, Camera
from match_frames import generate_match
from desc import Descriptor, Point
import matplotlib.pyplot as plt

def calibrate_image(image):
    return cv2.resize(image, (960, 540))

def triangulate_points(pose1, pose2, pts1, pts2):
    """
    Triangulate 3D points from two camera poses and corresponding 2D points.
    """
    points_3d = np.zeros((pts1.shape[0], 4))
    pose1_inv = np.linalg.inv(pose1)
    pose2_inv = np.linalg.inv(pose2)
    
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        A = np.array([
            pt1[0] * pose1_inv[2] - pose1_inv[0],
            pt1[1] * pose1_inv[2] - pose1_inv[1],
            pt2[0] * pose2_inv[2] - pose2_inv[0],
            pt2[1] * pose2_inv[2] - pose2_inv[1]
        ])
        _, _, vt = np.linalg.svd(A)
        points_3d[i] = vt[-1]
    
    return points_3d

def process_frame(image):
    """
    Process a single frame to perform SLAM.
    """
    image = calibrate_image(image)
    current_frame = Camera(desc_dict, image, K)
    if current_frame.id == 0:
        return

    prev_frame = desc_dict.frames[-1]
    older_frame = desc_dict.frames[-2]
    x1, x2, relative_pose = generate_match(prev_frame, older_frame)
    prev_frame.pose = np.dot(relative_pose, older_frame.pose)

    for i, idx in enumerate(x2):
        if older_frame.pts[idx] is not None:
            older_frame.pts[idx].add_observation(prev_frame, x1[i])

    points_3d = triangulate_points(prev_frame.pose, older_frame.pose, 
                                   prev_frame.key_pts[x1], older_frame.key_pts[x2])
    
    points_3d /= points_3d[:, 3:]  # Normalize homogeneous coordinates

    unmatched_points = np.array([prev_frame.pts[i] is None for i in x1])
    valid_points = (np.abs(points_3d[:, 3]) > 0.005) & (points_3d[:, 2] > 0) & unmatched_points

    for i, point in enumerate(points_3d):
        if not valid_points[i]:
            continue
        pt = Point(desc_dict, point)
        pt.add_observation(prev_frame, x1[i])
        pt.add_observation(older_frame, x2[i])

    for pt1, pt2 in zip(prev_frame.key_pts[x1], older_frame.key_pts[x2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        # How the optical flow is displayed.
        cv2.circle(image, (u1, v1), color=(0, 255, 0), radius=1)
        cv2.line(image, (u1, v1), (u2, v2), color=(255, 255, 0))

    # 3D point cloud visualization.
    desc_dict.update()
    return image

if __name__ == "__main__":
    # Constants
    video_frames_path = sys.argv[1]
    video_frames = np.load(video_frames_path)
    os.chdir("../../..")
    DATASET = "//00"

    # Initialize Calibration
    with open(os.path.abspath(os.curdir) + DATASET + "//calib.txt", 'r') as calib:
        calib_params = [float(item) for item in calib.readline().split(' ')[1:]]
        P = np.reshape(calib_params, (3,4))
        K = P[:3, :3]

    desc_dict = Descriptor(K)
    desc_dict.create_viewer()

    i = 0

    for frame in video_frames:
        i = i+1
        frame_resized = cv2.resize(frame, (720, 400))
        processed_frame = process_frame(frame_resized)
        #if(processed_frame is not None):
            #cv2.imshow("Frame", processed_frame)
        #else:
            #cv2.imshow("Frame", frame_resized)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        if(i == 50):
            break
    a, b = desc_dict.bundle_adjustment()
    plt.plot(a)
    plt.show()
    plt.plot(b.fun)
    plt.show()
    desc_dict.update_viewer()
    desc_dict.save_point_cloud()
    cv2.destroyAllWindows()