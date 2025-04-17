#!/usr/bin/env python3
"""
verify_fundamental.py: Load saved correspondences, compute the fundamental matrix via eight-point DLT,
and draw epipolar lines and point correspondences on both images for visual verification.

Usage:
    python verify_fundamental.py point_pairs.json img1.jpg img2.jpg [--save out.png]
"""
import argparse
import json
import cv2
import numpy as np

def normalize_points(pts):
    mean = pts.mean(axis=0)
    centered = pts - mean
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0,     0,              1.0]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_h = (T @ pts_h.T).T
    return pts_norm_h[:, :2], T


def eight_point_dlt(pts1, pts2):
    if pts1.shape[0] < 8:
        raise ValueError("Need at least 8 point pairs for DLT.")
    p1n, T1 = normalize_points(pts1)
    p2n, T2 = normalize_points(pts2)
    n = p1n.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = p1n[i]
        x2, y2 = p2n[i]
        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,      y1,      1]
    _, _, Vt = np.linalg.svd(A)
    F0 = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F0)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt2
    F = T2.T @ F_rank2 @ T1
    return F / F[2, 2] if abs(F[2, 2]) > 1e-8 else F


def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    # img1, img2: BGR images; pts1, pts2: Nx2 arrays
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # compute epilines in each image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    # base images
    out1 = img1.copy()
    out2 = img2.copy()
    # draw epipolar lines
    for (a, b, c) in lines1:
        if abs(b) > 1e-6:
            p0 = (0, int(-c / b)); p1 = (w1, int(-(c + a * w1) / b))
        else:
            x0 = int(-c / a); p0 = (x0, 0); p1 = (x0, h1)
        cv2.line(out1, p0, p1, (0, 255, 0), 1)
    for (a, b, c) in lines2:
        if abs(b) > 1e-6:
            p0 = (0, int(-c / b)); p1 = (w2, int(-(c + a * w2) / b))
        else:
            x0 = int(-c / a); p0 = (x0, 0); p1 = (x0, h2)
        cv2.line(out2, p0, p1, (0, 255, 0), 1)
    # overlay points with unique colors and opacity
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0)
    ]
    overlay1 = out1.copy()
    overlay2 = out2.copy()
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        color = colors[i % len(colors)]
        center1 = (int(pt1[0]), int(pt1[1]))
        center2 = (int(pt2[0]), int(pt2[1]))
        cv2.circle(overlay1, center1, 8, color, -1)
        cv2.circle(overlay2, center2, 8, color, -1)
    alpha = 0.8
    out1 = cv2.addWeighted(overlay1, alpha, out1, 1 - alpha, 0)
    out2 = cv2.addWeighted(overlay2, alpha, out2, 1 - alpha, 0)
    return out1, out2


def load_correspondences(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    pts1 = np.array([p['image1_point'] for p in data['point_pairs']], dtype=float)
    pts2 = np.array([p['image2_point'] for p in data['point_pairs']], dtype=float)
    return pts1, pts2


def main():
    parser = argparse.ArgumentParser(description='Verify Fundamental Matrix via epipolar lines')
    parser.add_argument('corr', help='JSON file with point_pairs')
    parser.add_argument('img1', help='First image')
    parser.add_argument('img2', help='Second image')
    parser.add_argument('--save', '-s', help='Save concatenated output to file')
    args = parser.parse_args()

    pts1, pts2 = load_correspondences(args.corr)
    print(f"Loaded {len(pts1)} correspondences.")
    F = eight_point_dlt(pts1, pts2)
    print("Fundamental matrix:")
    print(np.round(F, 6))

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    out1, out2 = draw_epipolar_lines(img1, img2, pts1, pts2, F)
    # Auto-save verification epipolar images
    cv2.imwrite("savedVerificationFundamentalMatrix1.png", out1)
    cv2.imwrite("savedVerificationFundamentalMatrix2.png", out2)
    print("Saved epipolar-line verification images to 'savedVerificationFundamentalMatrix1.png' and 'savedVerificationFundamentalMatrix2.png'")
    # concatenate for display
    concat = np.hstack((out1, out2))
    cv2.namedWindow('Epipolar Verification', cv2.WINDOW_NORMAL)
    cv2.imshow('Epipolar Verification', concat)
    if args.save:
        cv2.imwrite(args.save, concat)
        print(f"Saved visualization to {args.save}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
