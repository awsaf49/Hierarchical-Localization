#!/usr/bin/env python3
"""
visualize_triple_view.py: Script to visualize three images side by side with their 
corresponding points (without displaying matrices in the image).
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import sys
import os

sys.path.append("..")  # Add parent directory to path
from TripleViewFundamentalMatrix import eight_point_dlt, normalize_points

def visualize_triple_view(points_json, output_path=None):
    """
    Visualize three images side by side with corresponding points.
    
    Args:
        points_json: Path to JSON file with triplet correspondences
        output_path: Path to save the visualization image (optional)
    """
    # Load point triplets
    with open(points_json, 'r') as f:
        data = json.load(f)
    
    # Get image paths
    img_a_path = data['image_a']
    img_b_path = data['image_b']
    img_c_path = data['image_c']
    
    # Load images
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    img_c = cv2.imread(img_c_path)
    
    if img_a is None or img_b is None or img_c is None:
        print("Error: Could not load one or more images")
        return
    
    # Convert BGR to RGB for matplotlib
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    img_c_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    
    # Extract point correspondences
    pts_a = np.array([triplet['image_a_point'] for triplet in data['point_triplets']], dtype=float)
    pts_b = np.array([triplet['image_b_point'] for triplet in data['point_triplets']], dtype=float)
    pts_c = np.array([triplet['image_c_point'] for triplet in data['point_triplets']], dtype=float)
    
    # Compute fundamental matrices (for reference only, not displayed in image)
    F_ab = eight_point_dlt(pts_a, pts_b)
    F_cb = eight_point_dlt(pts_c, pts_b)
    
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
    
    # Print errors to console but don't include in visualization
    print(f"A→B mean epipolar error: {errors_ab.mean():.2f} px")
    print(f"C→B mean epipolar error: {errors_cb.mean():.2f} px")
    
    # Create figure for visualization with only the three images
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Display images
    axes[0].imshow(img_a_rgb)
    axes[1].imshow(img_b_rgb)
    axes[2].imshow(img_c_rgb)
    
    # Plot corresponding points with the same color
    for i in range(len(pts_a)):
        color = plt.cm.tab10(i % 10)
        
        # Plot points
        axes[0].plot(pts_a[i, 0], pts_a[i, 1], 'o', color=color, markersize=8)
        axes[1].plot(pts_b[i, 0], pts_b[i, 1], 'o', color=color, markersize=8)
        axes[2].plot(pts_c[i, 0], pts_c[i, 1], 'o', color=color, markersize=8)
        
        # Add point index labels
        axes[0].text(pts_a[i, 0] + 10, pts_a[i, 1] + 10, str(i+1), color=color, fontsize=12)
        axes[1].text(pts_b[i, 0] + 10, pts_b[i, 1] + 10, str(i+1), color=color, fontsize=12)
        axes[2].text(pts_c[i, 0] + 10, pts_c[i, 1] + 10, str(i+1), color=color, fontsize=12)
        
        # Draw lines connecting corresponding points
        con1 = ConnectionPatch(xyA=(pts_a[i, 0], pts_a[i, 1]), xyB=(pts_b[i, 0], pts_b[i, 1]),
                              coordsA="data", coordsB="data",
                              axesA=axes[0], axesB=axes[1], color=color, linestyle='--', alpha=0.5)
        con2 = ConnectionPatch(xyA=(pts_b[i, 0], pts_b[i, 1]), xyB=(pts_c[i, 0], pts_c[i, 1]),
                              coordsA="data", coordsB="data",
                              axesA=axes[1], axesB=axes[2], color=color, linestyle='--', alpha=0.5)
        
        fig.add_artist(con1)
        fig.add_artist(con2)
    
    # Set titles and turn off axes
    axes[0].set_title(f"Image A: {os.path.basename(img_a_path)}", fontsize=12)
    axes[1].set_title(f"Image B: {os.path.basename(img_b_path)}", fontsize=12)
    axes[2].set_title(f"Image C: {os.path.basename(img_c_path)}", fontsize=12)
    
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    
    # Add a main title for the figure
    fig.suptitle("Triple View Geometry: Corresponding Points Across Three Images", fontsize=16)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize triple view geometry with corresponding points")
    parser.add_argument("--points", default="../triple_view_points.json", 
                        help="JSON file with point triplets")
    parser.add_argument("--output", default="../triple_view_visualization.png",
                        help="Output path for visualization")
    
    args = parser.parse_args()
    
    visualize_triple_view(args.points, args.output)