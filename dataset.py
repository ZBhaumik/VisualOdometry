import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.chdir("..")
DATASET = "//" + "00"
path = os.path.abspath(os.curdir) + DATASET + "//image_0"
image_paths = os.listdir(path)
images = np.array([cv2.imread(path + "//" + image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths])
print(np.array(images))
np.save('images00.npy', np.array([[1,2,3],[4,5,6]]))