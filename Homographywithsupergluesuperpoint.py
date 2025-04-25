import argparse
import json
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from SuperGluePretrainedNetwork.models.matching import Matching


def compute_homography_dlt(src_points, dst_points):
    """
    Compute homography matrix H using Direct Linear Transform (DLT).
    dst_point ~ H * src_point
    """
    if len(src_points) < 4:
        raise ValueError("At least 4 matched point pairs are required")
    n = len(src_points)
    A = np.zeros((2 * n, 9))
    for i, (x, y) in enumerate(src_points):
        x_p, y_p = dst_points[i]
        A[2*i]   = [x, y, 1, 0, 0, 0, -x * x_p, -y * x_p, -x_p]
        A[2*i+1] = [0, 0, 0, x, y, 1, -x * y_p, -y * y_p, -y_p]
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, pt):
    p = np.array([pt[0], pt[1], 1.0])
    p2 = H @ p
    return [p2[0] / p2[2], p2[1] / p2[2]]


def visualize_homography_mapping(img1_path, img2_path,
                                 src_pts, dst_pts, mapped_pts,
                                 test_src=None, test_dst=None, test_mapped=None):
    # Load and convert images to RGB
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    ax1.imshow(img1_rgb); ax1.set_title('Image 1')
    ax2.imshow(img2_rgb); ax2.set_title('Image 2')

    # Plot source points as small red dots
    for p in src_pts:
        ax1.add_patch(Circle((p[0], p[1]), 2, color='r', fill=True))

    # Plot destination points as small blue dots
    for p in dst_pts:
        ax2.add_patch(Circle((p[0], p[1]), 2, color='b', fill=True))

    # Plot mapped points as small green dots
    for p in mapped_pts:
        ax2.plot(p[0], p[1], 'g.', markersize=4)

    # Optional test points
    if test_src is not None and test_dst is not None and test_mapped is not None:
        for p in test_src:
            ax1.plot(p[0], p[1], 'm.', markersize=4)
        for p in test_dst:
            ax2.plot(p[0], p[1], 'm.', markersize=4)
        for p in test_mapped:
            ax2.plot(p[0], p[1], 'y.', markersize=4)

    plt.tight_layout()
    plt.show()


def get_superglue_matches(img1_path, img2_path, device, config):
    # Read images in grayscale
    img0 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        raise FileNotFoundError(f"Cannot load images {img1_path}, {img2_path}")

    # Normalize and convert to torch tensors
    tensor0 = torch.from_numpy(img0.astype('float32') / 255.).unsqueeze(0).unsqueeze(0).to(device)
    tensor1 = torch.from_numpy(img1.astype('float32') / 255.).unsqueeze(0).unsqueeze(0).to(device)

    matcher = Matching(config).eval().to(device)
    data = {'image0': tensor0, 'image1': tensor1}
    with torch.no_grad():
        pred = matcher(data)

    # extract match info
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches0 = pred['matches0'][0].cpu().numpy()
    mconf   = pred['matching_scores0'][0].cpu().numpy()
    valid = matches0 > -1
    src_pts = kpts0[valid]
    dst_pts = kpts1[matches0[valid].astype(int)]
    confidences = mconf[valid]
    return src_pts, dst_pts, confidences


def main():
    parser = argparse.ArgumentParser(description='Compute homography using SuperPoint+SuperGlue')
    parser.add_argument('image1', help='Path to the first image')
    parser.add_argument('image2', help='Path to the second image')
    parser.add_argument('--match-threshold', type=float, default=0.2,
                        help='SuperGlue matching confidence threshold')
    parser.add_argument('--max-seed', type=int, default=4096,
                        help='Maximum keypoints to detect')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': args.max_seed
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': args.match_threshold
        }
    }

    # get matches
    src_pts, dst_pts, _ = get_superglue_matches(
        args.image1, args.image2, device, config)

    # split into train/test
    n = len(src_pts)
    n_train = max(int(0.7 * n), 4)
    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    src_train = src_pts[train_idx]
    dst_train = dst_pts[train_idx]
    src_test  = src_pts[test_idx]
    dst_test  = dst_pts[test_idx]

    # compute homography
    H = compute_homography_dlt(src_train, dst_train)
    print("Homography H:\n", H)

    # compute errors
    errors = [np.linalg.norm(np.array(apply_homography(H, p)) - q)
              for p, q in zip(src_train, dst_train)]
    print(f"Avg train error: {np.mean(errors):.2f}px")

    # visualize
    mapped_train = [apply_homography(H, p) for p in src_train]
    mapped_test  = [apply_homography(H, p) for p in src_test]

    visualize_homography_mapping(
        args.image1, args.image2,
        src_train, dst_train, mapped_train,
        src_test, dst_test, mapped_test
    )

if __name__ == '__main__':
    main()
