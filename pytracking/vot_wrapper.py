import vot
import sys
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pytracking.tracker.segm import Segm
from pytracking.parameter.segm import default_params as vot_params


def rect_to_poly(rect):
    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[0] + rect[2]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]
    x3 = rect[0]
    y3 = rect[1] + rect[3]
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def parse_sequence_name(image_path):
    idx = image_path.find('/color/')
    return image_path[idx - image_path[:idx][::-1].find('/'):idx], idx

def parse_frame_name(image_path, idx):
    frame_name = image_path[idx + len('/color/'):]
    return frame_name[:frame_name.find('.')]

# MAIN
handle = vot.VOT("polygon")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

params = vot_params.parameters()

gt_rect = [round(selection.points[0].x, 2), round(selection.points[0].y, 2),
           round(selection.points[1].x, 2), round(selection.points[1].y, 2),
           round(selection.points[2].x, 2), round(selection.points[2].y, 2),
           round(selection.points[3].x, 2), round(selection.points[3].y, 2)]

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

sequence_name, idx_ = parse_sequence_name(imagefile)
frame_name = parse_frame_name(imagefile, idx_)

params.masks_save_path = ''
params.save_mask = False

tracker = Segm(params)

# tell the sequence name to the tracker (to save segmentation masks to the disk)
tracker.sequence_name = sequence_name
tracker.frame_name = frame_name

tracker.initialize(image, gt_rect)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    # tell the frame name to the tracker (to save segmentation masks to the disk)
    frame_name = parse_frame_name(imagefile, idx_)
    tracker.frame_name = frame_name

    prediction = tracker.track(image)

    if len(prediction) == 4:
        prediction = rect_to_poly(prediction)

    pred_poly = vot.Polygon([vot.Point(prediction[0], prediction[1]),
                             vot.Point(prediction[2], prediction[3]),
                             vot.Point(prediction[4], prediction[5]),
                             vot.Point(prediction[6], prediction[7])])

    handle.report(pred_poly)
