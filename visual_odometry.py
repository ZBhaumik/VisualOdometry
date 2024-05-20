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
        #self.images = [cv2.imread(path + "//" + image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

        # Initialize Calibration
        with open(os.path.abspath(os.curdir) + DATASET + "//calib.txt", 'r') as calib:
            calib_params = [float(item) for item in calib.readline().split(' ')[1:]]
            self.P = np.reshape(calib_params, (3,4))
            self.K = self.P[:3, :3]
            print(self.P)
            print(self.K)

def main():
    VO = MonoVO("00")

if __name__ == "__main__":
    main()