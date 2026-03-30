import numpy as np
from pathlib import Path
import cv2
import os

from detection.canny import Canny
from detection.first_order import FirstOrder
from detection.second_order import SecondOrder
from detection.deriche import DericheDetector
from detection.sub_pixel import SubPixel
from error import EdgeEvaluation


def label_txt_to_outline_mask(txt_path, image_shape):
    values = np.fromstring(Path(txt_path).read_text().strip(), sep=' ', dtype=np.float64)
    polygon_xy = values[1:].reshape(-1, 2)

    h, w = image_shape[:2]
    polygon_px = np.stack(
        [polygon_xy[:, 0] * w, polygon_xy[:, 1] * h],
        axis=1,
    ).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [polygon_px], isClosed=True, color=255, thickness=2)
    return mask

if __name__ == "__main__":
    