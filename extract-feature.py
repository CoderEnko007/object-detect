from sklearn.externals import joblib
from skimage.feature import hog
from data.config import config
import argparse
import glob
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pospath", required=True, help="Path to Positive images")
ap.add_argument("-n", "--negpath", required=True, help="Path to Negative images")
args = vars(ap.parse_args())

pos_image_path = args["pospath"]
neg_image_path = args["negpath"]

if not os.path.isdir(config.pos_feature_path):
    os.makedirs(config.pos_feature_path)

if not os.path.isdir(config.neg_feature_path):
    os.makedirs(config.neg_feature_path)

print(config.pixels_per_cell)
for path in glob.glob(os.path.join(args["pospath"], "*")):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd = hog(image, orientations=config.orientations, pixels_per_cell=config.pixels_per_cell,
             cells_per_block=config.cells_per_block, visualise=config.visualise)
    fd_name = os.path.split(path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(config.pos_feature_path, fd_name)
    joblib.dump(fd, fd_path)

for path in glob.glob(os.path.join(args["negpath"], "*")):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd = hog(image, orientations=config.orientations, pixels_per_cell=config.pixels_per_cell,
             cells_per_block=config.cells_per_block, visualise=config.visualise)
    fd_name = os.path.split(path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(config.neg_feature_path, fd_name)
    joblib.dump(fd, fd_path)
