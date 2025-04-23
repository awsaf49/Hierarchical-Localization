#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import torch

# SuperPoint and SuperGlue import
try:
    from superpoint import SuperPointFrontend as SuperPoint
    from SuperGluePretrainedNetwork.models.matching import Matching as SuperGlue
except ImportError as e:
    raise ImportError("Check that 'superpoint.py' and SuperGlue are correctly placed.\n" + str(e))


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


def extract_matches(img0, img1, device, sp_conf, sg_conf):
    sp = SuperPoint(
        weights_path=sp_conf['weights_path'],
        nms_dist=sp_conf['nms_dist'],
        conf_thresh=sp_conf['conf_thresh'],
        nn_thresh=sp_conf['nn_thresh'],
        cuda=(device == 'cuda')
    )
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t0 = torch.from_numpy(g0)[None, None].to(device)
    t1 = torch.from_numpy(g1)[None, None].to(device)

    pts0, desc0, _ = sp.run(g0)
    pts1, desc1, _ = sp.run(g1)

    kp0 = torch.from_numpy(pts0[:2].T).float().to(device)[None]
    kp1 = torch.from_numpy(pts1[:2].T).float().to(device)[None]
    desc0_t = torch.from_numpy(desc0).float().to(device)[None]
    desc1_t = torch.from_numpy(desc1).float().to(device)[None]
    scores0 = torch.from_numpy(pts0[2]).float().to(device)[None]
    scores1 = torch.from_numpy(pts1[2]).float().to(device)[None]

    sg = SuperGlue({'superpoint': sp_conf, 'superglue': sg_conf}).eval().to(device)

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
    pts0_list, pts1_list = [], []
    for i, m in enumerate(matches):
        if m > -1:
            pts0_list.append(pts0[:2, i])
            pts1_list.append(pts1[:2, m])
    return np.array(pts0_list), np.array(pts1_list)


def draw_epipolar_lines_and_keypoints(img_b_path, pts_b1, F_ab, pts_b2, F_cb):
    img = cv2.imread(img_b_path)
    h, w = img.shape[:2]

    def draw_lines(img, lines, color):
        for (a, b, c) in lines:
            if abs(b) > 1e-6:
                pt1 = (0, int(-c / b))
                pt2 = (w, int(-(c + a * w) / b))
            else:
                x = int(-c / a)
                pt1 = (x, 0)
                pt2 = (x, h)
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

    # Draw epipolar lines
    lines_ab = cv2.computeCorrespondEpilines(pts_b1.reshape(-1, 1, 2), 2, F_ab).reshape(-1, 3)
    draw_lines(img, lines_ab, (0, 255, 0))  # Green

    lines_cb = cv2.computeCorrespondEpilines(pts_b2.reshape(-1, 1, 2), 2, F_cb).reshape(-1, 3)
    draw_lines(img, lines_cb, (255, 0, 255))  # Purple

    # Draw keypoints
    for pt in pts_b1:
        cv2.circle(img, tuple(pt.astype(int)), 4, (0, 0, 255), -1)  # Red (A→B)
    for pt in pts_b2:
        cv2.circle(img, tuple(pt.astype(int)), 4, (255, 0, 0), -1)  # Blue (C→B)

    # Legends
    cv2.putText(img, "Green lines / Red dots: A→B", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Purple lines / Blue dots: C→B", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description='Multi-view epipolar visualization with keypoints and labels')
    parser.add_argument('image_a')
    parser.add_argument('image_b')
    parser.add_argument('image_c')
    parser.add_argument('--draw', '-d', action='store_true')
    parser.add_argument('--save', type=str, default=None, help='Optional path to save output image')
    args = parser.parse_args()

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

    img_a = cv2.imread(args.image_a)
    img_b = cv2.imread(args.image_b)
    img_c = cv2.imread(args.image_c)

    pts_a, pts_b1 = extract_matches(img_a, img_b, device, sp_conf, sg_conf)
    pts_c, pts_b2 = extract_matches(img_c, img_b, device, sp_conf, sg_conf)

    print(f"A→B matches: {len(pts_b1)}")
    print(f"C→B matches: {len(pts_b2)}")

    if len(pts_b1) < 8 or len(pts_b2) < 8:
        print("Not enough matches to compute both F matrices.")
        return

    F_ab = eight_point_dlt(pts_a, pts_b1)
    F_cb = eight_point_dlt(pts_c, pts_b2)

    print("\nFundamental Matrix A→B:")
    print(np.round(F_ab, 6))
    print("\nFundamental Matrix C→B:")
    print(np.round(F_cb, 6))

    if args.draw:
        result = draw_epipolar_lines_and_keypoints(args.image_b, pts_b1, F_ab, pts_b2, F_cb)
        cv2.imshow("Epipolar Lines on B", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if args.save:
            cv2.imwrite(args.save, result)
            print(f"Saved result to: {args.save}")


if __name__ == '__main__':
    main()
