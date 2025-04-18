#!/usr/bin/env python3
"""
TripleViewFundamentalMatrix.py: A comprehensive tool for computing and verifying 
fundamental matrices across three images A, B, and C.

This script:
1. Allows selecting corresponding points across all 3 images simultaneously
2. Computes F matrices between A→B and C→B using 8-point DLT with normalization
3. Visualizes epipolar lines from both A and C onto image B
4. Provides a verification mechanism using at least 5 points

Usage:
    python TripleViewFundamentalMatrix.py --image_a PATH --image_b PATH --image_c PATH
    python TripleViewFundamentalMatrix.py --verify --image_a PATH --image_b PATH --image_c PATH --points PATH
"""

import argparse
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class TriplePointSelector:
    def __init__(self, img_a_path, img_b_path, img_c_path, output_path=None):
        """Initialize point selector for three images."""
        # Load images
        self.img_a = cv2.imread(img_a_path)
        self.img_b = cv2.imread(img_b_path)
        self.img_c = cv2.imread(img_c_path)
        
        if self.img_a is None or self.img_b is None or self.img_c is None:
            raise ValueError("Could not load one or more images.")
            
        # Convert BGR to RGB for matplotlib
        self.img_a_rgb = cv2.cvtColor(self.img_a, cv2.COLOR_BGR2RGB)
        self.img_b_rgb = cv2.cvtColor(self.img_b, cv2.COLOR_BGR2RGB)
        self.img_c_rgb = cv2.cvtColor(self.img_c, cv2.COLOR_BGR2RGB)
        
        self.img_a_path = img_a_path
        self.img_b_path = img_b_path
        self.img_c_path = img_c_path
        
        # Default output path
        self.output_path = output_path or "triple_view_points.json"
        
        # Store point triplets: [[x_a, y_a, x_b, y_b, x_c, y_c], ...]
        self.point_triplets = []
        
        # Load existing points if available
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                if (data.get('image_a') == os.path.abspath(self.img_a_path) and
                    data.get('image_b') == os.path.abspath(self.img_b_path) and
                    data.get('image_c') == os.path.abspath(self.img_c_path)):
                    for trip in data.get('point_triplets', []):
                        pt_a = trip['image_a_point']
                        pt_b = trip['image_b_point']
                        pt_c = trip['image_c_point']
                        self.point_triplets.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], pt_c[0], pt_c[1]])
            except Exception as e:
                print(f"Error loading existing points: {e}")
        
        # Track selection state (0=image A, 1=image B, 2=image C)
        self.select_state = 0
        self.current_triplet = [None, None, None]
        
        # Set up the visualization
        self.setup_display()
    
    def setup_display(self):
        """Set up the matplotlib figure and axes for point selection."""
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(2, 3)
        
        # Image A (top left)
        self.ax_a = self.fig.add_subplot(gs[0, 0])
        self.ax_a.imshow(self.img_a_rgb)
        self.ax_a.set_title(f"Image A: {os.path.basename(self.img_a_path)}")
        self.ax_a.axis('off')
        
        # Image B (top center)
        self.ax_b = self.fig.add_subplot(gs[0, 1])
        self.ax_b.imshow(self.img_b_rgb)
        self.ax_b.set_title(f"Image B: {os.path.basename(self.img_b_path)}")
        self.ax_b.axis('off')
        
        # Image C (top right)
        self.ax_c = self.fig.add_subplot(gs[0, 2])
        self.ax_c.imshow(self.img_c_rgb)
        self.ax_c.set_title(f"Image C: {os.path.basename(self.img_c_path)}")
        self.ax_c.axis('off')
        
        # Bottom for instructions and buttons
        self.ax_inst = self.fig.add_subplot(gs[1, :])
        self.ax_inst.axis('off')
        
        # Add buttons
        ax_save = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_clear = plt.axes([0.5, 0.05, 0.1, 0.075])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear Last')
        
        self.btn_save.on_clicked(self.save_points)
        self.btn_clear.on_clicked(self.clear_last)
        
        # Add figure title and instructions
        self.fig.suptitle(
            "Select corresponding points across all three images in order: A → B → C", 
            fontsize=14
        )
        
        # Status text at the bottom
        self.instruction_text = self.ax_inst.text(
            0.5, 0.7, 
            f"Select a point in Image {'A' if self.select_state == 0 else 'B' if self.select_state == 1 else 'C'} | "
            f"Triplets: {len(self.point_triplets)}",
            ha='center', fontsize=12
        )
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.redraw()
    
    def on_click(self, event):
        """Handle mouse clicks for point selection."""
        if event.inaxes is None:
            return
            
        x, y = event.xdata, event.ydata
        
        if self.select_state == 0 and event.inaxes == self.ax_a:
            # Image A
            self.current_triplet[0] = [x, y]
            self.select_state = 1
            self.ax_a.plot(x, y, 'ro')
            self.fig.canvas.draw_idle()
            
        elif self.select_state == 1 and event.inaxes == self.ax_b:
            # Image B
            self.current_triplet[1] = [x, y]
            self.select_state = 2
            self.ax_b.plot(x, y, 'ro')
            self.fig.canvas.draw_idle()
            
        elif self.select_state == 2 and event.inaxes == self.ax_c:
            # Image C
            self.current_triplet[2] = [x, y]
            
            # Add the complete triplet
            idx = len(self.point_triplets) + 1
            self.point_triplets.append([
                self.current_triplet[0][0], self.current_triplet[0][1],  # A
                self.current_triplet[1][0], self.current_triplet[1][1],  # B
                self.current_triplet[2][0], self.current_triplet[2][1],  # C
            ])
            
            # Reset for next triplet
            self.select_state = 0
            self.current_triplet = [None, None, None]
            
            # Update visualization
            self.redraw()
            
            # Update instruction text
            self.instruction_text.set_text(
                f"Select a point in Image {'A' if self.select_state == 0 else 'B' if self.select_state == 1 else 'C'} | "
                f"Triplets: {len(self.point_triplets)}"
            )
            self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 'enter':
            self.save_points(None)
            print(f"Saved {len(self.point_triplets)} triplets to {self.output_path}")
        elif event.key == 'backspace':
            self.clear_last(None)
    
    def clear_last(self, event):
        """Remove the last added triplet or reset the current selection."""
        if self.select_state > 0:
            # Reset the current partial triplet
            self.select_state = 0
            self.current_triplet = [None, None, None]
            self.redraw()
        elif self.point_triplets:
            # Remove the last complete triplet
            self.point_triplets.pop()
            self.redraw()
        
        # Update instruction text
        self.instruction_text.set_text(
            f"Select a point in Image {'A' if self.select_state == 0 else 'B' if self.select_state == 1 else 'C'} | "
            f"Triplets: {len(self.point_triplets)}"
        )
        self.fig.canvas.draw_idle()
    
    def redraw(self):
        """Redraw all points on the three images."""
        # Clear the axes
        self.ax_a.clear()
        self.ax_b.clear()
        self.ax_c.clear()
        
        # Redisplay the images
        self.ax_a.imshow(self.img_a_rgb)
        self.ax_a.set_title(f"Image A: {os.path.basename(self.img_a_path)}")
        self.ax_a.axis('off')
        
        self.ax_b.imshow(self.img_b_rgb)
        self.ax_b.set_title(f"Image B: {os.path.basename(self.img_b_path)}")
        self.ax_b.axis('off')
        
        self.ax_c.imshow(self.img_c_rgb)
        self.ax_c.set_title(f"Image C: {os.path.basename(self.img_c_path)}")
        self.ax_c.axis('off')
        
        # Draw all triplets with consistent colors
        for i, (x_a, y_a, x_b, y_b, x_c, y_c) in enumerate(self.point_triplets, start=1):
            color = plt.cm.tab10(i % 10)
            
            self.ax_a.plot(x_a, y_a, 'o', color=color)
            self.ax_b.plot(x_b, y_b, 'o', color=color)
            self.ax_c.plot(x_c, y_c, 'o', color=color)
            
            # Add point index labels
            self.ax_a.text(x_a + 5, y_a + 5, str(i), color=color)
            self.ax_b.text(x_b + 5, y_b + 5, str(i), color=color)
            self.ax_c.text(x_c + 5, y_c + 5, str(i), color=color)
        
        # If there's a partial triplet being selected, draw it too
        if self.current_triplet[0] is not None:
            self.ax_a.plot(self.current_triplet[0][0], self.current_triplet[0][1], 'ro')
        if self.current_triplet[1] is not None:
            self.ax_b.plot(self.current_triplet[1][0], self.current_triplet[1][1], 'ro')
        
        # Update the instruction text
        self.instruction_text.set_text(
            f"Select a point in Image {'A' if self.select_state == 0 else 'B' if self.select_state == 1 else 'C'} | "
            f"Triplets: {len(self.point_triplets)}"
        )
        
        self.fig.canvas.draw_idle()
    
    def save_points(self, event):
        """Save the selected point triplets to a JSON file."""
        data = {
            'image_a': os.path.abspath(self.img_a_path),
            'image_b': os.path.abspath(self.img_b_path),
            'image_c': os.path.abspath(self.img_c_path),
            'point_triplets': []
        }
        
        for x_a, y_a, x_b, y_b, x_c, y_c in self.point_triplets:
            data['point_triplets'].append({
                'image_a_point': [int(x_a), int(y_a)],
                'image_b_point': [int(x_b), int(y_b)],
                'image_c_point': [int(x_c), int(y_c)]
            })
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        if event is not None:  # Only print if called from button
            print(f"Saved {len(self.point_triplets)} triplets to {self.output_path}")
    
    def run(self):
        """Run the interactive point selector."""
        plt.show()


def normalize_points(pts):
    """
    Normalize a set of 2D points by translating to the origin and 
    scaling to achieve a mean distance of sqrt(2) from the origin.
    
    Args:
        pts: Nx2 array of 2D points
        
    Returns:
        normalized_pts: Nx2 array of normalized points
        T: 3x3 transformation matrix used for normalization
    """
    # Compute centroid
    mean = pts.mean(axis=0)
    
    # Center the points
    centered = pts - mean
    
    # Compute mean distance from origin
    distances = np.linalg.norm(centered, axis=1)
    mean_dist = distances.mean()
    
    # Scale factor to achieve average distance of sqrt(2)
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    # Create transformation matrix
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation: convert to homogeneous coordinates, transform, and convert back
    pts_homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    normalized_pts_homog = (T @ pts_homog.T).T
    normalized_pts = normalized_pts_homog[:, :2]
    
    return normalized_pts, T


def eight_point_dlt(pts1, pts2):
    """
    Compute the fundamental matrix using the normalized 8-point algorithm with DLT.
    
    Args:
        pts1: Nx2 array of points in the first image
        pts2: Nx2 array of corresponding points in the second image
        
    Returns:
        F: 3x3 fundamental matrix
    """
    if len(pts1) < 8:
        raise ValueError("At least 8 point correspondences required for 8-point algorithm")
    
    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Create the constraint matrix A
    n = pts1_norm.shape[0]
    A = np.zeros((n, 9))
    
    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        
        A[i] = [
            x2*x1, x2*y1, x2,
            y2*x1, y2*y1, y2,
            x1, y1, 1
        ]
    
    # Solve for f using SVD
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]  # Last row of Vt = last column of V = eigenvector of smallest eigenvalue
    F_norm = f.reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0  # Set smallest singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt
    
    # Denormalize to get F in the original coordinate system
    F = T2.T @ F_rank2 @ T1
    
    # Scale F so that last element is 1 (if not too close to 0)
    if abs(F[2, 2]) > 1e-8:
        F = F / F[2, 2]
    
    return F


def draw_epilines(img, pts_src, F, which='AB', color=(0, 255, 0), alpha=0.6):
    """
    Draw epipolar lines on an image from source points and a fundamental matrix.
    
    Args:
        img: Image on which to draw epipolar lines
        pts_src: Nx2 array of source points
        F: 3x3 fundamental matrix
        which: String indicating the direction ('AB' or 'CB')
        color: Color for the epipolar lines
        alpha: Transparency for the overlay
        
    Returns:
        overlay: Image with epipolar lines overlaid
    """
    overlay = img.copy()
    h, w = img.shape[:2]
    
    # Reshape points for cv2
    pts_src_cv = pts_src.reshape(-1, 1, 2)
    
    # Compute epipolar lines
    # which=1 means lines in second image from points in first
    # which=2 means lines in first image from points in second
    which_cv = 1 if which == 'AB' else 2
    lines = cv2.computeCorrespondEpilines(pts_src_cv, which_cv, F).reshape(-1, 3)
    
    # Draw each line
    for line in lines:
        a, b, c = line
        
        # Calculate two points on the line
        if abs(b) > 1e-6:  # Line is not vertical
            y0, y1 = 0, h
            x0 = int(-c / a) if abs(a) > 1e-6 else 0
            x1 = int(-(c + b * h) / a) if abs(a) > 1e-6 else x0
        else:  # Line is vertical
            x0 = int(-c / a) if abs(a) > 1e-6 else 0
            x1 = x0
            y0, y1 = 0, h
        
        # Draw the line
        cv2.line(overlay, (x0, y0), (x1, y1), color, 2)
    
    # Create transparent overlay
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def verify_fundamental_matrices(img_a_path, img_b_path, img_c_path, points_file):
    """
    Verify fundamental matrices by computing them and drawing epipolar lines.
    
    Args:
        img_a_path: Path to image A
        img_b_path: Path to image B
        img_c_path: Path to image C
        points_file: Path to JSON file with triplet correspondences
    """
    # Load images
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    img_c = cv2.imread(img_c_path)
    
    if img_a is None or img_b is None or img_c is None:
        print("Error: Could not load one or more images")
        return
    
    # Load point triplets
    with open(points_file, 'r') as f:
        data = json.load(f)
    
    # Extract point correspondences
    pts_a = np.array([triplet['image_a_point'] for triplet in data['point_triplets']], dtype=float)
    pts_b = np.array([triplet['image_b_point'] for triplet in data['point_triplets']], dtype=float)
    pts_c = np.array([triplet['image_c_point'] for triplet in data['point_triplets']], dtype=float)
    
    n_points = len(pts_a)
    
    print(f"Loaded {n_points} point triplets")
    
    if n_points < 8:
        print("Error: At least 8 correspondences required for fundamental matrix computation")
        return
    
    # Compute fundamental matrices
    F_ab = eight_point_dlt(pts_a, pts_b)
    F_cb = eight_point_dlt(pts_c, pts_b)
    
    print("Fundamental matrix A→B:")
    print(np.round(F_ab, 6))
    print("\nFundamental matrix C→B:")
    print(np.round(F_cb, 6))
    
    # Calculate epipolar errors
    def compute_epipolar_error(pts1, pts2, F):
        errors = []
        for i in range(pts1.shape[0]):
            p1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
            p2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])
            # Epipolar line in image 2
            line = F @ p1
            # Distance from point to line
            err = abs(p2.dot(line)) / np.sqrt(line[0]**2 + line[1]**2)
            errors.append(err)
        return np.array(errors)
    
    errors_ab = compute_epipolar_error(pts_a, pts_b, F_ab)
    errors_cb = compute_epipolar_error(pts_c, pts_b, F_cb)
    
    print(f"A→B mean epipolar error: {errors_ab.mean():.2f} px, max: {errors_ab.max():.2f} px")
    print(f"C→B mean epipolar error: {errors_cb.mean():.2f} px, max: {errors_cb.max():.2f} px")
    
    # Draw epipolar lines from A onto B (green)
    img_b_with_ab_lines = draw_epilines(img_b, pts_a, F_ab, which='AB', color=(0, 255, 0), alpha=0.7)
    
    # Draw epipolar lines from C onto B (blue)
    img_b_with_both_lines = draw_epilines(img_b_with_ab_lines, pts_c, F_cb, which='CB', color=(255, 0, 0), alpha=0.7)
    
    # Draw the corresponding points on image B
    for i, pt in enumerate(pts_b):
        x, y = int(pt[0]), int(pt[1])
        color = plt.cm.tab10(i % 10)
        # Convert matplotlib color to OpenCV color (RGB -> BGR)
        cv_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.circle(img_b_with_both_lines, (x, y), 8, cv_color, -1)
        cv2.putText(img_b_with_both_lines, str(i + 1), (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save verification image
    output_path = "triple_view_verification.png"
    cv2.imwrite(output_path, img_b_with_both_lines)
    print(f"Saved verification image to {output_path}")
    
    # Display verification image
    cv2.namedWindow("Fundamental Matrix Verification", cv2.WINDOW_NORMAL)
    cv2.imshow("Fundamental Matrix Verification", img_b_with_both_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Triple-View Fundamental Matrix Computation and Verification")
    parser.add_argument("--image_a", help="Path to image A")
    parser.add_argument("--image_b", help="Path to image B")
    parser.add_argument("--image_c", help="Path to image C")
    parser.add_argument("--points", default="triple_view_points.json", help="JSON file for point triplets")
    parser.add_argument("--verify", action="store_true", help="Verify existing fundamental matrices")
    
    args = parser.parse_args()
    
    if not (args.image_a and args.image_b and args.image_c):
        parser.error("All three images (--image_a, --image_b, --image_c) are required")
    
    if args.verify:
        verify_fundamental_matrices(args.image_a, args.image_b, args.image_c, args.points)
    else:
        print(f"Triple-View Point Selection: {args.image_a}, {args.image_b}, {args.image_c}")
        selector = TriplePointSelector(args.image_a, args.image_b, args.image_c, args.points)
        selector.run()
        print(f"Saved point triplets to {args.points}")


if __name__ == "__main__":
    main()