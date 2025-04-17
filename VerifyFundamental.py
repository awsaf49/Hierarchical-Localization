#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse


def load_correspondences(points_file):
    """
    Load point correspondences from JSON file.

    Returns:
        pts1, pts2: Nx2 numpy arrays of points
    """
    with open(points_file, 'r') as f:
        data = json.load(f)
    pts1 = np.array([p['image1_point'] for p in data['point_pairs']], dtype=np.float32)
    pts2 = np.array([p['image2_point'] for p in data['point_pairs']], dtype=np.float32)
    return pts1, pts2


def compute_fundamental(pts1, pts2, ransac_thresh=3.0, confidence=0.99):
    """
    Estimate the fundamental matrix with RANSAC.

    Returns:
        F: 3x3 fundamental matrix
        mask: inlier mask from RANSAC
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, confidence)
    return F, mask


def compute_epipolar_errors(pts1, pts2, F):
    """
    Compute distances of pts2 to epipolar lines induced by pts1.

    Returns:
        errors: array of distances (pixels)
    """
    errors = []
    for p1, p2 in zip(pts1, pts2):
        x1 = np.array([p1[0], p1[1], 1.0])
        line2 = F.dot(x1)       # epipolar line in image2: a x + b y + c = 0
        a, b, c = line2
        err = abs(a * p2[0] + b * p2[1] + c) / np.sqrt(a*a + b*b)
        errors.append(err)
    return np.array(errors)


def draw_epipolar_lines(img1_path, img2_path, pts1, pts2, F):
    """
    Display images with epipolar lines and points at original resolution.
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError("Could not load images for epipolar display.")

    # compute epilines
    pts1_cv = pts1.reshape(-1,1,2)
    pts2_cv = pts2.reshape(-1,1,2)
    lines2 = cv2.computeCorrespondEpilines(pts1_cv, 1, F).reshape(-1,3)
    lines1 = cv2.computeCorrespondEpilines(pts2_cv, 2, F).reshape(-1,3)

    def draw_lines(img, lines, pts):
        h, w = img.shape[:2]
        for (a, b, c), pt in zip(lines, pts):
            if abs(b) > 1e-6:
                x0, y0 = 0, int(-c / b)
                x1, y1 = w, int(-(c + a * w) / b)
            else:
                x0, y0 = int(-c / a), 0
                x1, y1 = int(-c / a), h
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        return img

    vis1 = draw_lines(img1.copy(), lines1, pts1)
    vis2 = draw_lines(img2.copy(), lines2, pts2)

    # Create adjustable windows matching image size
    win1 = 'Epilines on Image 1'
    win2 = 'Epilines on Image 2'
    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win1, vis1.shape[1], vis1.shape[0])
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win2, vis2.shape[1], vis2.shape[0])

    cv2.imshow(win1, vis1)
    cv2.imshow(win2, vis2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Verify fundamental matrix with epipolar error and full-resolution visualization"
    )
    parser.add_argument('--points', default="point_pairs_fundamental.json", help="Path to JSON file of point correspondences")
    parser.add_argument('--image1', help="First image path")
    parser.add_argument('--image2', help="Second image path")
    parser.add_argument('--draw', '-d', action='store_true',
                        help="Draw epipolar lines and points on both images at full size")
    args = parser.parse_args()

    pts1, pts2 = load_correspondences(args.points)
    F, mask = compute_fundamental(pts1, pts2)
    if F is None:
        print("Failed to estimate fundamental matrix.")
        return

    print("Estimated Fundamental Matrix:")
    print(F)
    inliers = int(mask.ravel().sum())
    print(f"Inliers: {inliers} / {len(pts1)}")

    errors = compute_epipolar_errors(pts1, pts2, F)
    print(f"Mean epipolar error: {errors.mean():.2f} pixels")
    print(f"Max epipolar error: {errors.max():.2f} pixels")

    if args.draw:
        draw_epipolar_lines(args.image1, args.image2, pts1, pts2, F)

if __name__ == '__main__':
    main()
