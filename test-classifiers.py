from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from skimage.feature import hog
from data.config import config
from nms import non_max_suppression_fast
import numpy as np
import argparse
import imutils
import cv2


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to test image", required=True)
ap.add_argument("-d", "--downscale", help="Downscale ratio", default=1.5, type=int)
ap.add_argument("-v", "--visualize", help="Visualize the sliding window", action="store_true")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
downscale = args["downscale"]
visualize = args["visualize"]

clf = joblib.load(config.model_path)
detections = []
scale = 0

ratio = image.shape[1] / 200
orig = image.copy()
scaled_width = int(image.shape[1] / ratio)
image = imutils.resize(image, width=scaled_width if ratio > 1 else image.shape[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
for scaled_image in pyramid_gaussian(gray, downscale=downscale):
    current_detection = []
    if scaled_image.shape[0] < config.min_window_size[1] or scaled_image.shape[1] < config.min_window_size[0]:
        break
    for (x, y, window) in sliding_window(scaled_image, config.step_size, config.min_window_size):
        if window.shape[0] != config.min_window_size[1] or window.shape[1] != config.min_window_size[0]:
            continue
        fd = hog(window, orientations=config.orientations, pixels_per_cell=config.pixels_per_cell,
                 cells_per_block=config.cells_per_block, visualise=config.visualise)
        pred = clf.predict([fd])
        if pred == 1:
            print("Location:(%s, %s)" % (x, y))
            print("Confidence:%s" % clf.decision_function([fd]))
            detections.append((x, y, clf.decision_function([fd]),
                               int(config.min_window_size[0] * (downscale ** scale)),
                               int(config.min_window_size[1] * (downscale ** scale))))
            current_detection.append(detections[-1])
        if visualize:
            clone = scaled_image.copy()
            for x1, y1, _, _, _ in current_detection:
                cv2.rectangle(clone, (x1, y1), (x1 + window.shape[1], y1 + window.shape[0]), 0, 2)
            cv2.rectangle(clone, (x, y), (x + window.shape[1], y + window.shape[0]), 255, 2)
            cv2.imshow("Sliding window processing", clone)
            cv2.waitKey(30)
    scale += 1

clone = image.copy()
for (x, y, _, w, h) in detections:
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("image before NMS", clone)
cv2.waitKey(0)

boxes = np.array([(x, y, x + w, y + h, d) for (x, y, d, w, h) in detections])
detections = non_max_suppression_fast(boxes, 0.2)
if ratio > 1:
    detections = detections[:4] * int(ratio)
for (x1, y1, x2, y2, d) in detections:
    print(d)
    if d > 0.5:
        cv2.rectangle(orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
cv2.imshow("image after NMS", imutils.resize(orig, width=800 if orig.shape[1] > 800 else orig.shape[1]))
cv2.waitKey(0)
