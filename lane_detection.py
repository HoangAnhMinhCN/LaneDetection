#!/usr/bin/env python3
"""
Lane Detection v2 (more accurate)
Pipeline: resize -> white/yellow colour mask + Gaussian blur + Canny
          -> adaptive Region of Interest -> Hough -> robust left/right fit
          -> vanishing-point based line length -> temporal smoothing -> overlay.

Usage:
    pip install opencv-python numpy
    python lane_detection.py input.mp4 output.mp4 [--show]
    python lane_detection.py road.jpg result.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np


class LaneDetector:
    def __init__(self, proc_width=960, smooth=0.9, length=0.75):
        self.proc_width = proc_width   # frames are resized to this width for detection
        self.smooth = smooth           # temporal smoothing (0 = off)
        self.length = length           # line length: fraction of the way from the
                                       # vanishing point down to the bottom (smaller = shorter)
        self.left = None               # (a, b): x = a*y + b
        self.right = None
        self.vp_y = None               # smoothed horizon / vanishing point y

    # ---- 1. paint mask: thin bright stripes (white) or yellow ---------------
    @staticmethod
    def colour_mask(frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        # top-hat keeps thin features brighter than their surroundings (lane paint)
        # and ignores big bright areas (sky) and haze that lowers absolute brightness
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, k)
        white = ((tophat > 20) & (l > 100) & (hls[:, :, 2] < 110)).astype(np.uint8) * 255
        yellow = cv2.inRange(hls, (12, 90, 90), (38, 255, 255))
        return cv2.bitwise_or(white, yellow)

    #Guassian blur + Canny edge detection, restricted to painted pixels
    # ---- 2. edges, restricted to painted pixels ---------------------------
    def detect_edges(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        # auto Canny thresholds from the ROAD half only (bright sky would skew them)
        med = np.median(gray[gray.shape[0] // 2:])
        low = float(np.clip(0.5 * med, 30, 80))
        edges = cv2.Canny(gray, int(low), int(np.clip(3 * low, 90, 200)))
        mask = cv2.dilate(self.colour_mask(blur), np.ones((7, 7), np.uint8))
        return cv2.bitwise_and(edges, mask)
    
    # Region of interest
    # ---- 3. adaptive region of interest -----------------------------------
    def roi_polygon(self, h, w):
        top = (self.vp_y + 0.04 * h) if self.vp_y is not None else 0.62 * h
        top = int(np.clip(top, 0.45 * h, 0.9 * h))
        return np.array([[(0, h), (int(0.40 * w), top),
                          (int(0.60 * w), top), (w, h)]], dtype=np.int32)

    def region_of_interest(self, edges):
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_polygon(*edges.shape), 255)
        return cv2.bitwise_and(edges, mask)

    #Hough transform to detect lines
    # ---- 4. Hough ---------------------------------------------------------
    @staticmethod
    def detect_lines(img):
        h = img.shape[0]
        return cv2.HoughLinesP(img, 1, np.pi / 180, threshold=25,
                               minLineLength=int(0.05 * h), maxLineGap=int(0.04 * h))

    # ---- 5. robust left/right estimation ----------------------------------
    def estimate(self, lines, h, w):
        left, right = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                if y1 == y2:
                    continue
                a = (x2 - x1) / (y2 - y1)          # dx/dy (stable for steep lines)
                if not 0.35 < abs(a) < 2.2:         # reject near-vertical / near-flat
                    continue
                b = x1 - a * y1
                length = np.hypot(x2 - x1, y2 - y1)
                x_bot = a * h + b                   # where the line hits the bottom
                # a = dx/dy: going UP the image the left lane drifts right (a < 0),
                # the right lane drifts left (a > 0)
                if a < 0 and x_bot < 0.5 * w:
                    left.append((a, b, length, x_bot))
                elif a > 0 and x_bot > 0.5 * w:
                    right.append((a, b, length, x_bot))

        self.left = self._fit(self.left, left, h)
        self.right = self._fit(self.right, right, h)

        # vanishing point = intersection of the two lines
        if self.left and self.right and abs(self.left[0] - self.right[0]) > 1e-3:
            y = (self.right[1] - self.left[1]) / (self.left[0] - self.right[0])
            if 0.3 * h < y < 0.95 * h:
                self.vp_y = y if self.vp_y is None else 0.9 * self.vp_y + 0.1 * y

    def _fit(self, prev, segs, h):
        if not segs:
            return prev
        arr = np.array(segs)
        # outlier rejection: drop segments far from the weighted median bottom-x
        order = np.argsort(arr[:, 3])
        cum = np.cumsum(arr[order, 2])
        med = arr[order, 3][np.searchsorted(cum, cum[-1] / 2)]
        keep = np.abs(arr[:, 3] - med) < 0.08 * h * 2
        if prev is not None:                        # stay close to previous estimate
            prev_bot = prev[0] * h + prev[1]
            keep &= np.abs(arr[:, 3] - prev_bot) < 0.25 * h * 2
        if keep.sum() == 0:
            return prev
        arr = arr[keep]
        w = arr[:, 2]
        fit = (np.average(arr[:, 0], weights=w), np.average(arr[:, 1], weights=w))
        if prev is None:
            return fit
        s = self.smooth
        return (s * prev[0] + (1 - s) * fit[0], s * prev[1] + (1 - s) * fit[1])

    # ---- 6. draw ----------------------------------------------------------
    def _segment(self, fit, h):
        if fit is None:
            return None
        vp = self.vp_y if self.vp_y is not None else 0.62 * h
        y_top = vp + (1 - self.length) * (h - vp)
        y_top = int(min(y_top, 0.97 * h))
        a, b = fit
        return int(a * h + b), h, int(a * y_top + b), y_top

    def draw(self, frame, scale):
        h = frame.shape[0]
        hp = int(h * scale)
        overlay = np.zeros_like(frame)
        L, R = self._segment(self.left, hp), self._segment(self.right, hp)

        def up(seg):  # back to full-resolution coordinates
            return [int(v / scale) for v in seg]

        if L and R:
            l, r = up(L), up(R)
            cv2.fillPoly(overlay, [np.array([(l[0], l[1]), (l[2], l[3]),
                                             (r[2], r[3]), (r[0], r[1])], np.int32)],
                         (0, 255, 0))
        thick = max(3, h // 150)
        for seg in (L, R):
            if seg:
                s = up(seg)
                cv2.line(overlay, (s[0], s[1]), (s[2], s[3]), (0, 0, 255), thick)
        return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

    def process(self, frame):
        scale = min(1.0, self.proc_width / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        roi = self.region_of_interest(self.detect_edges(small))
        self.estimate(self.detect_lines(roi), *small.shape[:2])
        return self.draw(frame, scale)


def process_image(src, dst):
    img = cv2.imread(src)
    if img is None:
        sys.exit(f"Cannot read image: {src}")
    det = LaneDetector(smooth=0.0)
    det.process(img)          # 1st pass finds the vanishing point
    cv2.imwrite(dst, det.process(img))
    print(f"Saved: {dst}")


def process_video(src, dst, show=False):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    det = LaneDetector()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.process(frame)
        out.write(res)
        n += 1
        if n % 30 == 0:
            print(f"Processing {n}/{total if total > 0 else '?'}")
        if show:
            cv2.imshow("Lane Detection", cv2.resize(res, (960, int(960 * h / w))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"Done: {dst}")


def main():
    p = argparse.ArgumentParser(description="Lane detection")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    if os.path.splitext(a.input)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        process_image(a.input, a.output)
    else:
        process_video(a.input, a.output, a.show)


if __name__ == "__main__":
    main()
