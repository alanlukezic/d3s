import cv2
import numpy as np
import copy


def sum_in(integral_img, y1, x1, y2, x2):
    return float(integral_img[y2 + 1, x2 + 1] - integral_img[y2 + 1, x1] - integral_img[y1, x2 + 1] + integral_img[y1, x1])

def make_opt_step(mask_integral, y1, x1, y2, x2, alpha, min_factor, n_total):
    if not (y1 < mask_integral.shape[0] and x1 < mask_integral.shape[1] and y2 >= 0 and x2 >= 0):
        return -1
    N1 = sum_in(mask_integral, y1, x1, y2, x2)

    if N1 / n_total < min_factor:
        return -1

    intersection = N1
    A = float((y2 - y1 + 1) * (x2 - x1 + 1))
    union_ = alpha * (A - N1) + N1 + (n_total - N1)
    if (union_ < 1e-3):
        return -1
    iou = intersection / union_

    return iou

def fit_aa_box(mask, rotated=True):
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    contour = contours[np.argmax(cnt_area)]
    polygon = contour.reshape(-1, 2)

    # start with the min-max rectangle
    xx1 = max(np.min(polygon[:, 0]), 0)
    yy1 = max(np.min(polygon[:, 1]), 0)
    xx2 = min(np.max(polygon[:, 0]), mask.shape[1] - 1)
    yy2 = min(np.max(polygon[:, 1]), mask.shape[0] - 1)

    n_total = float(np.sum(mask))
    min_factor = 0.9
    alpha = 0.25

    best = -1
    best_rect = [xx1, yy1, xx2, yy2]
    if rotated:
        mask_integral = cv2.integral(mask)

        x1 = copy.deepcopy(xx1)
        y1 = copy.deepcopy(yy1)
        x2 = copy.deepcopy(xx2)
        y2 = copy.deepcopy(yy2)
        while True:
            changed = False

            iou_ = make_opt_step(mask_integral, y1 + 1, x1, y2, x2, alpha, min_factor, n_total)
            if iou_ > best:
                y1 += 1
                best = iou_
                changed = True
            iou_ = make_opt_step(mask_integral, y1, x1 + 1, y2, x2, alpha, min_factor, n_total)
            if iou_ > best:
                x1 += 1
                best = iou_
                changed = True
            iou_ = make_opt_step(mask_integral, y1, x1, y2 - 1, x2, alpha, min_factor, n_total)
            if iou_ > best:
                y2 -= 1
                best = iou_
                changed = True
            iou_ = make_opt_step(mask_integral, y1, x1, y2, x2 - 1, alpha, min_factor, n_total)
            if iou_ > best:
                x2 -= 1
                best = iou_
                changed = True

            if not changed:
                break

        best_rect = [x1, y1, x2, y2]

    # output is in the form: [x0, y0, x1, y1]
    return best_rect


def fit_bbox_to_mask(mask, rotated=True):
    # binarize mask (just in case if it has not been binarized yet)
    target_mask = (mask > 0.3)
    target_mask = target_mask.astype(np.uint8)

    # get contours from mask
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]

    if len(contours) != 0 and np.max(cnt_area) > 50:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)

        if rotated:
            ellipseBox = cv2.fitEllipse(polygon)
            # get the center of the ellipse and the angle
            angle = ellipseBox[-1]
            center = np.array(ellipseBox[0])
            axes = np.array(ellipseBox[1])

            if axes[0] > mask.shape[0] * 0.6 or axes[1] > mask.shape[1] * 0.6:
                # fit of an ellipse might went wrong - fit min area rectangle
                ellipseBox = cv2.minAreaRect(polygon)
                angle = ellipseBox[-1]
                center = np.array(ellipseBox[0])

            # calculate rotation matrix and rotate mask
            R = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
            rotated_mask = cv2.warpAffine(target_mask, R, (target_mask.shape[1], target_mask.shape[0]))

            # fit axis-aligned rectagnle to the rotated mask
            aa_box = fit_aa_box(rotated_mask)
            aa_poly = np.array([[aa_box[0], aa_box[1]], [aa_box[2], aa_box[1]], [aa_box[2], aa_box[3]], [aa_box[0], aa_box[3]]])

            # transform estimated rectangle back (inverse rotation)
            R_inv = cv2.invertAffineTransform(R)
            one = np.ones([aa_poly.shape[0], 3, 1])
            one[:, :2, :] = aa_poly.reshape(-1, 2, 1)
            output = np.matmul(R_inv, one).reshape(-1, 2)

        else:
            # no need to estimate rotation of the bbox - just fit axis-aligned bbox to the mask
            aa_box = fit_aa_box(target_mask, rotated=False)
            output = np.array([[aa_box[0], aa_box[1]], [aa_box[2], aa_box[1]], [aa_box[2], aa_box[3]], [aa_box[0], aa_box[3]]])

    else:  # empty mask
        output = None

    return output
