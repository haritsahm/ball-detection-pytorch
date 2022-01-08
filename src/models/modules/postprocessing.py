from typing import List

import cv2
import numpy as np

def fcnnv2_post_processs(heatmaps: np.ndarray) -> np.ndarray:
    l_candidates = []
    for im in heatmaps:
        heatmap_int8 = (im * 255).astype(np.uint8)
        _,thresh = cv2.threshold(heatmap_int8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Compute cnt and find max val
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_val = 0
        max_cx, max_cy = -1, -1
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            roi = im[y:y+h, x:x+w]
            rad = np.maximum(w, h)
            values = roi[roi != 0].sum()
            if values > max_val:
                max_val = values
            
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                max_cx, max_cy = cx, cy
        l_candidates.append((max_cx, max_cy,rad))

    return np.array(l_candidates)
