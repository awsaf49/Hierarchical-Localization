#!/usr/bin/env python3
"""
Compute the fundamental matrix automatically using SuperPoint + SuperGlue matches,
then verify via normalized eight-point DLT and epipolar line visualization.
Requires superpoint.py (SuperPointFrontend) and superglue.py (Matching) in your project directory.
"""
import argparse
import cv2
import numpy as np
import torch

# Import SuperPoint frontend and SuperGlue matcher
try:
    from superpoint import SuperPointFrontend as SuperPoint
    from superglue import Matching as SuperGlue
except ImportError as e:
    raise ImportError(
        "Ensure 'superpoint.py' defines SuperPointFrontend and 'superglue.py' defines Matching.\n" + str(e)
    )


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
        raise ValueError("Need at least 8 matches for DLT.")
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)
    A = np.zeros((len(pts1_n), 9))
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1_n, pts2_n)):
        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,      y1,      1]
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F_norm)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt2
    F = T2.T @ F_rank2 @ T1
    if abs(F[2, 2]) > 1e-8:
        F = F / F[2, 2]
    return F


def compute_epipolar_errors(pts1, pts2, F):
    errs = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        a, b, c = F @ np.array([x1, y1, 1.0])
        errs.append(abs(a * x2 + b * y2 + c) / np.hypot(a, b))
    return np.array(errs)


def draw_epipolar_lines(img0_path, img1_path, pts0, pts1, F):
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    pts0_cv = pts0.reshape(-1, 1, 2)
    pts1_cv = pts1.reshape(-1, 1, 2)
    lines1 = cv2.computeCorrespondEpilines(pts0_cv, 1, F).reshape(-1, 3)
    lines0 = cv2.computeCorrespondEpilines(pts1_cv, 2, F).reshape(-1, 3)
    def draw(img, lines, pts):
        h, w = img.shape[:2]
        for (a, b, c), pt in zip(lines, pts):
            if abs(b) > 1e-6:
                p0 = (0, int(-c / b)); p1 = (w, int(-(c + a * w) / b))
            else:
                x0 = int(-c / a); p0 = (x0, 0); p1 = (x0, h)
            cv2.line(img, p0, p1, (0, 255, 0), 1)
            cv2.circle(img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        return img
    vis0 = draw(img0.copy(), lines0, pts0)
    vis1 = draw(img1.copy(), lines1, pts1)
    for name, vis in [('Epilines0', vis0), ('Epilines1', vis1)]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        h, w = vis.shape[:2]
        cv2.resizeWindow(name, w, h)
        cv2.imshow(name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_matches(img0, img1, device, sp_conf, sg_conf):
    # Initialize SuperPoint
    sp = SuperPoint(
        weights_path=sp_conf['weights_path'],
        nms_dist=sp_conf['nms_dist'],
        conf_thresh=sp_conf['conf_thresh'],
        nn_thresh=sp_conf['nn_thresh'],
        cuda=(device == 'cuda')
    )
    # Prepare grayscale images
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t0 = torch.from_numpy(g0)[None, None].to(device)
    t1 = torch.from_numpy(g1)[None, None].to(device)

    # Detect with SuperPoint
    pts0, desc0, _ = sp.run(g0)
    pts1, desc1, _ = sp.run(g1)

    # Convert to tensors for SuperGlue
    kp0 = torch.from_numpy(pts0[:2].T).float().to(device)[None]  # [1, N, 2]
    kp1 = torch.from_numpy(pts1[:2].T).float().to(device)[None]
    # Descriptors: [D, N] from SuperPoint, convert to [1, D, N]
    desc0_t = torch.from_numpy(desc0).float().to(device)[None]
    desc1_t = torch.from_numpy(desc1).float().to(device)[None]
    # Scores/confidences: [N] -> [1, N]
    scores0 = torch.from_numpy(pts0[2]).float().to(device)[None]
    scores1 = torch.from_numpy(pts1[2]).float().to(device)[None]

    # Initialize SuperGlue matcher
    sg = SuperGlue(
        {'superpoint': sp_conf, 'superglue': sg_conf}
    ).eval().to(device)

    # Prepare input dict
    data = {
        'image0': t0,
        'image1': t1,
        'keypoints0': kp0,
        'keypoints1': kp1,
        'descriptors0': desc0_t,
        'descriptors1': desc1_t,
        'scores0': scores0,
        'scores1': scores1
    }
    with torch.no_grad():
        pred = sg(data)
    matches = pred['matches0'][0].cpu().numpy()

    # Extract matched points
    pts0_list, pts1_list = [], []
    for i, m in enumerate(matches):
        if m > -1:
            pts0_list.append(pts0[:2, i])
            pts1_list.append(pts1[:2, m])
    return np.array(pts0_list), np.array(pts1_list)


def main():
    parser = argparse.ArgumentParser(
        description='Auto fundamental via SuperPoint + SuperGlue + DLT'
    )
    parser.add_argument('image0')
    parser.add_argument('image1')
    parser.add_argument('--draw', '-d', action='store_true')
    args = parser.parse_args()

    img0 = cv2.imread(args.image0)
    img1 = cv2.imread(args.image1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp_conf = {
        'weights_path': 'superpoint_v1.pth',
        'nms_dist': 4,
        'conf_thresh': 0.015,
        'nn_thresh': 0.7
    }
    sg_conf = {
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }

    pts0, pts1 = extract_matches(img0, img1, device, sp_conf, sg_conf)
    print(f"Found {len(pts0)} matches.")
    if len(pts0) < 8:
        print("Not enough matches to compute F.")
        return

    F = eight_point_dlt(pts0, pts1)
    print("Fundamental matrix via DLT:")
    print(np.round(F, 6))
    errs = compute_epipolar_errors(pts0, pts1, F)
    print(f"Mean epipolar error: {errs.mean():.2f} px, Max: {errs.max():.2f} px")

    if args.draw:
        draw_epipolar_lines(args.image0, args.image1, pts0, pts1, F)

if __name__ == '__main__':
    main()

# $env:PYTHONPATH += ";C:\Users\pushp\OneDrive\Documents\Haystac\ECE281\Hierarchical-Localization\SuperGluePretrainedNetwork"
# python FundamentalSuperpointSuperglue.py img_1.JPG img_2.JPG --draw
# $env:PYTHONPATH += ";C:\Users\pushp\OneDrive\Documents\Haystac\ECE281\Hierarchical-Localization\SuperPointPretrainedNetwork"
# python FundamentalSuperpointSuperglue.py IMG1.jpg IMG2.jpg --draw
