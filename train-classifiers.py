from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from data.config import config
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--posfeat", help="Path to the positive feature", required=True)
ap.add_argument("-n", "--negfeat", help="Path to the negative feature", required=True)
args = vars(ap.parse_args())

pos_feature_path = args["posfeat"]
neg_feature_path = args["negfeat"]

fds = []
labels = []

for path in glob.glob(os.path.join(pos_feature_path, "*.feat")):
    fd = joblib.load(path)
    fds.append(fd)
    labels.append(1)

for path in glob.glob(os.path.join(neg_feature_path, "*.feat")):
    fd = joblib.load(path)
    fds.append(fd)
    labels.append(0)

clf = LinearSVC()
clf.fit(fds, labels)
if not os.path.isdir(os.path.split(config.model_path)[0]):
    os.makedirs(os.path.split(config.model_path)[0])
joblib.dump(clf, config.model_path)
