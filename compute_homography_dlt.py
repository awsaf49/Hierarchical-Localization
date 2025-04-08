import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def compute_homography_dlt(src_points, dst_points):
    """
    Compute homography matrix H using Direct Linear Transform (DLT) such that:
    dst_point ~ H * src_point

    Args:
        src_points: Points from source image (nx2 array)
        dst_points: Corresponding points in destination image (nx2 array)

    Returns:
        H: 3x3 homography matrix
    """
    if len(src_points) < 4:
        raise ValueError("At least 4 point pairs are needed to compute homography")

    # Number of point pairs
    n = len(src_points)

    # Create the coefficient matrix A of size 2n x 9
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]

        # Each point pair gives us 2 equations
        # For x_prime:
        # x_prime*(h31*x + h32*y + h33) = h11*x + h12*y + h13
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime, -x_prime]

        # For y_prime:
        # y_prime*(h31*x + h32*y + h33) = h21*x + h22*y + h23
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime, -y_prime]

    # Solve Ah = 0 using SVD
    # The solution is the eigenvector of A^T*A with the smallest eigenvalue
    _, _, V = np.linalg.svd(A)

    # The solution is the last row of V
    h = V[-1]

    # Reshape into 3x3 matrix
    H = h.reshape(3, 3)

    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]

    return H


def apply_homography(H, point):
    """
    Apply homography transformation to a point

    Args:
        H: 3x3 homography matrix
        point: 2D point [x, y]

    Returns:
        Transformed point [x', y']
    """
    # Convert to homogeneous coordinates
    p = np.array([point[0], point[1], 1])

    # Apply transformation
    p_prime = H @ p

    # Convert back from homogeneous coordinates
    return [p_prime[0] / p_prime[2], p_prime[1] / p_prime[2]]


def visualize_homography_mapping(
    image1_path, image2_path, src_points, dst_points, mapped_points, 
    test_src_points=None, test_dst_points=None, test_mapped_points=None
):
    """
    Visualize the original points and the mapped points on images
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        src_points: Source points used for homography estimation
        dst_points: Destination points used for homography estimation
        mapped_points: Source points after applying homography
        test_src_points: Test source points (not used in homography computation)
        test_dst_points: Test destination points (ground truth)
        test_mapped_points: Test source points after applying homography
    """
    try:
        import cv2

        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            print(f"Could not load images from {image1_path} and {image2_path}")
            return

        # Convert from BGR to RGB for matplotlib
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Display images
        ax1.imshow(img1)
        ax2.imshow(img2)

        # Plot training points on image 1 (circle markers)
        for point in src_points:
            ax1.add_patch(Circle((point[0], point[1]), 10, color="red", fill=True))

        # Plot training points on image 2 (circle markers)
        for point in dst_points:
            ax2.add_patch(Circle((point[0], point[1]), 10, color="blue", fill=True))

        # Plot mapped training points on image 2 (plus markers)
        for point in mapped_points:
            ax2.plot(point[0], point[1], 'g+', markersize=12, markeredgewidth=2)
            
        # Plot test points if provided
        if test_src_points is not None and test_dst_points is not None and test_mapped_points is not None:
            # Test points on image 1 (diamond markers)
            for point in test_src_points:
                ax1.plot(point[0], point[1], 'md', markersize=10)
                
            # Ground truth test points on image 2 (diamond markers)
            for point in test_dst_points:
                ax2.plot(point[0], point[1], 'md', markersize=10)
                
            # Mapped test points on image 2 (x markers)
            for point in test_mapped_points:
                ax2.plot(point[0], point[1], 'yx', markersize=12, markeredgewidth=2)

        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Training Source Points'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Training Destination Points'),
            plt.Line2D([0], [0], marker='+', color='g', markersize=10, label='Mapped Training Points'),
        ]
        
        if test_src_points is not None:
            handles.extend([
                plt.Line2D([0], [0], marker='d', color='m', markersize=10, label='Test Points'),
                plt.Line2D([0], [0], marker='x', color='y', markersize=10, label='Mapped Test Points')
            ])
            
        ax2.legend(handles=handles, loc='upper right')

        ax1.set_title('Image 1 with Source Points')
        ax2.set_title('Image 2 with Destination and Mapped Points')

        plt.tight_layout()
        plt.savefig("homography_mapping.png")
        print("Visualization saved as 'homography_mapping.png'")
    except ImportError:
        print("OpenCV not available, skipping visualization")
    except Exception as e:
        print(f"Error in visualization: {e}")


def main():
    # Load point pairs from JSON
    with open("point_pairs.json", "r") as f:
        data = json.load(f)

    # Extract image paths and point pairs
    image1_path = data["image1"]
    image2_path = data["image2"]
    
    # Get all point pairs
    all_src_points = np.array([pair["image1_point"] for pair in data["point_pairs"]])
    all_dst_points = np.array([pair["image2_point"] for pair in data["point_pairs"]])
    
    # Split data into training (for homography estimation) and testing sets
    # Use 70% of points for training, 30% for testing
    n_points = len(all_src_points)
    n_train = int(0.7 * n_points)
    
    # Ensure we have at least 4 points for DLT
    n_train = max(n_train, 4)
    
    # Random indices for training
    np.random.seed(42)  # For reproducibility
    train_indices = np.random.choice(n_points, n_train, replace=False)
    test_indices = np.array([i for i in range(n_points) if i not in train_indices])
    
    # Split the data
    src_points = all_src_points[train_indices]
    dst_points = all_dst_points[train_indices]
    
    test_src_points = all_src_points[test_indices]
    test_dst_points = all_dst_points[test_indices]
    
    print(f"Using {len(src_points)} points for homography estimation and {len(test_src_points)} points for testing")

    # Compute homography matrix using DLT
    H = compute_homography_dlt(src_points, dst_points)

    print("Computed Homography Matrix:")
    print(H)
    print("\nVerification of point mappings (Training Set):")

    # Map all training points and compute errors
    mapped_points = []
    total_error = 0

    for i in range(len(src_points)):
        src_point = src_points[i]
        dst_point = dst_points[i]

        # Apply homography to source point
        mapped_point = apply_homography(H, src_point)
        mapped_points.append(mapped_point)

        # Calculate error
        error = np.sqrt(
            (mapped_point[0] - dst_point[0]) ** 2
            + (mapped_point[1] - dst_point[1]) ** 2
        )

        print(f"Point {i+1}:")
        print(f"  Source: {src_point}")
        print(f"  Actual destination: {dst_point}")
        print(f"  Mapped destination: [{mapped_point[0]:.2f}, {mapped_point[1]:.2f}]")
        print(f"  Error: {error:.2f} pixels")

        total_error += error

    print(f"\nAverage training error: {total_error/len(src_points):.2f} pixels")
    
    # Test with points not used for homography estimation
    print("\nTesting with held-out points:")
    test_mapped_points = []
    test_total_error = 0
    
    for i in range(len(test_src_points)):
        src_point = test_src_points[i]
        dst_point = test_dst_points[i]
        
        # Apply homography to test source point
        mapped_point = apply_homography(H, src_point)
        test_mapped_points.append(mapped_point)
        
        # Calculate error
        error = np.sqrt(
            (mapped_point[0] - dst_point[0]) ** 2
            + (mapped_point[1] - dst_point[1]) ** 2
        )
        
        print(f"Test Point {i+1}:")
        print(f"  Source: {src_point}")
        print(f"  Actual destination: {dst_point}")
        print(f"  Mapped destination: [{mapped_point[0]:.2f}, {mapped_point[1]:.2f}]")
        print(f"  Error: {error:.2f} pixels")
        
        test_total_error += error
    
    if len(test_src_points) > 0:
        print(f"\nAverage test error: {test_total_error/len(test_src_points):.2f} pixels")

    # Visualize the results
    try:
        visualize_homography_mapping(
            image1_path, image2_path, 
            src_points, dst_points, mapped_points,
            test_src_points, test_dst_points, test_mapped_points
        )
    except Exception as e:
        print(f"Error in visualization: {e}")

if __name__ == "__main__":
    main()
