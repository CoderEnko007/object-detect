import argparse
import glob
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

for imagePath in glob.glob(os.path.join(args["image"], "*")):
    print(imagePath)
    image = cv2.imread(imagePath)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # k = cv2.waitKey(1) & 0xff
    # if k == ord('q'):
    #     break
