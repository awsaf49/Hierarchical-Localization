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
    image1_path, image2_path, src_points, dst_points, mapped_points
):
    """
    Visualize the original points and the mapped points on images
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

        # Plot original points on image 1
        for point in src_points:
            ax1.add_patch(Circle((point[0], point[1]), 5, color="red", fill=True))

        # Plot original points on image 2
        for point in dst_points:
            ax2.add_patch(Circle((point[0], point[1]), 5, color="blue", fill=True))

        # Plot mapped points on image 2
        for point in mapped_points:
            ax2.add_patch(
                Circle((point[0], point[1]), 5, color="green", fill=False, linewidth=2)
            )

        ax1.set_title("Image 1 with Source Points")
        ax2.set_title(
            "Image 2 with Destination Points (blue) and Mapped Points (green)"
        )

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

    # Get point pairs
    src_points = np.array([pair["image1_point"] for pair in data["point_pairs"]])
    dst_points = np.array([pair["image2_point"] for pair in data["point_pairs"]])

    # Compute homography matrix using DLT
    H = compute_homography_dlt(src_points, dst_points)

    print("Computed Homography Matrix:")
    print(H)
    print("\nVerification of point mappings:")

    # Map all points and compute errors
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

    print(f"\nAverage error: {total_error/len(src_points):.2f} pixels")

    # Test with a new point (using the first point as test)
    test_point = src_points[0]
    print(f"\nTest with first point: {test_point}")
    mapped_test = apply_homography(H, test_point)
    actual_dest = dst_points[0]
    test_error = np.sqrt(
        (mapped_test[0] - actual_dest[0]) ** 2 + (mapped_test[1] - actual_dest[1]) ** 2
    )

    print(f"Mapped to: [{mapped_test[0]:.2f}, {mapped_test[1]:.2f}]")
    print(f"Actual destination: {actual_dest}")
    print(f"Error: {test_error:.2f} pixels")

    # Visualize the results
    try:
        visualize_homography_mapping(
            image1_path, image2_path, src_points, dst_points, mapped_points
        )
    except Exception as e:
        print(f"Error in visualization: {e}")


if __name__ == "__main__":
    main()
