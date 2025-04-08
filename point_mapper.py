import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os


class PointMapper:
    def __init__(self, img1_path, img2_path, output_path=None):
        """
        Initialize the point mapper with two images.

        Args:
            img1_path: Path to the first image
            img2_path: Path to the second image
            output_path: Path to save the point pairs (optional)
        """
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)

        if self.img1 is None or self.img2 is None:
            raise ValueError("Could not load one or both images.")

        # Convert BGR to RGB for matplotlib
        self.img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

        self.img1_path = img1_path
        self.img2_path = img2_path
        self.output_path = output_path if output_path else "point_pairs.json"

        # Store point pairs: [[x1, y1, x2, y2], ...]
        self.point_pairs = []

        # Load existing point pairs if the output file exists
        self.existing_data = None
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    self.existing_data = json.load(f)
                    # Check if the existing file contains the same image pair
                    if self.existing_data.get("image1") == os.path.abspath(
                        self.img1_path
                    ) and self.existing_data.get("image2") == os.path.abspath(
                        self.img2_path
                    ):
                        print(
                            f"Found existing point pairs for these images in {self.output_path}"
                        )
                        # Load existing point pairs into our current list
                        for pair in self.existing_data.get("point_pairs", []):
                            x1, y1 = pair["image1_point"]
                            x2, y2 = pair["image2_point"]
                            self.point_pairs.append([x1, y1, x2, y2])
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading existing point pairs: {e}")
                self.existing_data = None

        # Flag to track which image we're currently selecting a point in
        self.selecting_img1 = True
        self.current_pair = [None, None]

        # Setup the figure and connect events
        self.setup_display()

    def setup_display(self):
        """Set up the matplotlib figure and connect event handlers."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display the images
        self.ax1.imshow(self.img1_rgb)
        self.ax1.set_title(f"Image 1: {os.path.basename(self.img1_path)}")
        self.ax1.axis("off")

        self.ax2.imshow(self.img2_rgb)
        self.ax2.set_title(f"Image 2: {os.path.basename(self.img2_path)}")
        self.ax2.axis("off")

        # Text instructions
        self.fig.suptitle(
            "Click on corresponding points in both images\n"
            "Press 'Enter' to save points, 'Esc' to exit, 'Backspace' to delete last pair"
        )

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Display any existing point pairs
        if self.point_pairs:
            for i, (x1, y1, x2, y2) in enumerate(self.point_pairs):
                color = plt.cm.tab10((i + 1) % 10)

                # Plot points with the same color
                self.ax1.plot(x1, y1, "o", color=color, markersize=5)
                self.ax2.plot(x2, y2, "o", color=color, markersize=5)

                # Add labels with the point pair index
                self.ax1.text(x1 + 5, y1 + 5, str(i + 1), color=color)
                self.ax2.text(x2 + 5, y2 + 5, str(i + 1), color=color)

        # Create text for instructions
        self.instruction_text = self.fig.text(
            0.5,
            0.01,
            f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
            f"Pairs: {len(self.point_pairs)}",
            ha="center",
        )

    def on_click(self, event):
        """Handle mouse click events on the images."""
        # Ignore clicks outside the axes
        if event.inaxes is None:
            return

        # Get the coordinates of the click
        x, y = event.xdata, event.ydata

        if self.selecting_img1 and event.inaxes == self.ax1:
            # First point (in image 1)
            self.current_pair[0] = [x, y]
            self.selecting_img1 = False

            # Mark the point on image 1
            self.ax1.plot(x, y, "ro", markersize=5)
            self.fig.canvas.draw()

        elif not self.selecting_img1 and event.inaxes == self.ax2:
            # Second point (in image 2)
            self.current_pair[1] = [x, y]

            # Add the pair to our list
            point_pair = [
                self.current_pair[0][0],
                self.current_pair[0][1],
                self.current_pair[1][0],
                self.current_pair[1][1],
            ]
            self.point_pairs.append(point_pair)

            # Mark the point on image 2
            self.ax2.plot(x, y, "ro", markersize=5)

            # Draw a line connecting corresponding points with the same color
            pair_idx = len(self.point_pairs)
            color = plt.cm.tab10(pair_idx % 10)

            # Use the same color for both points in the pair
            self.ax1.plot(
                self.current_pair[0][0],
                self.current_pair[0][1],
                "o",
                color=color,
                markersize=5,
            )
            self.ax2.plot(x, y, "o", color=color, markersize=5)

            # Add a label with the point pair index
            self.ax1.text(
                self.current_pair[0][0] + 5,
                self.current_pair[0][1] + 5,
                str(pair_idx),
                color=color,
            )
            self.ax2.text(x + 5, y + 5, str(pair_idx), color=color)

            # Reset for the next pair
            self.selecting_img1 = True
            self.current_pair = [None, None]

            # Update the instruction text
            self.instruction_text.set_text(
                f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
                f"Pairs: {len(self.point_pairs)}"
            )
            self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == "enter":
            # Save the point pairs
            self.save_points()
            print(f"Saved {len(self.point_pairs)} point pairs to {self.output_path}")

        elif event.key == "escape":
            # Exit without saving
            plt.close(self.fig)

        elif event.key == "backspace":
            # Remove the last point pair
            if self.point_pairs:
                self.point_pairs.pop()
                # Redraw the figure to update the points
                self.redraw()

    def redraw(self):
        """Redraw the figure with current point pairs."""
        self.ax1.clear()
        self.ax2.clear()

        # Show the images again
        self.ax1.imshow(self.img1_rgb)
        self.ax1.set_title(f"Image 1: {os.path.basename(self.img1_path)}")
        self.ax1.axis("off")

        self.ax2.imshow(self.img2_rgb)
        self.ax2.set_title(f"Image 2: {os.path.basename(self.img2_path)}")
        self.ax2.axis("off")

        # Plot all remaining point pairs
        for i, (x1, y1, x2, y2) in enumerate(self.point_pairs):
            color = plt.cm.tab10((i + 1) % 10)

            # Plot points and add labels
            self.ax1.plot(x1, y1, "o", color=color, markersize=5)
            self.ax2.plot(x2, y2, "o", color=color, markersize=5)

            self.ax1.text(x1 + 5, y1 + 5, str(i + 1), color=color)
            self.ax2.text(x2 + 5, y2 + 5, str(i + 1), color=color)

        # Update the instruction text
        self.instruction_text.set_text(
            f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
            f"Pairs: {len(self.point_pairs)}"
        )

        self.fig.canvas.draw()

    def save_points(self):
        """Save the point pairs to a JSON file."""
        # Check if existing data file exists and load it
        existing_data = None
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = None

        # Format the data for output
        data = {
            "image1": os.path.abspath(self.img1_path),
            "image2": os.path.abspath(self.img2_path),
            "point_pairs": [],
        }

        # Add new point pairs from current session
        for x1, y1, x2, y2 in self.point_pairs:
            data["point_pairs"].append(
                {"image1_point": [int(x1), int(y1)], "image2_point": [int(x2), int(y2)]}
            )

        # If we have existing data for the same image pair, preserve those points too
        if existing_data:
            # Check if it's the same image pair
            if (
                existing_data.get("image1") == data["image1"]
                and existing_data.get("image2") == data["image2"]
            ):
                # Get only existing points that aren't duplicates of our new points
                existing_points = existing_data.get("point_pairs", [])
                new_points_set = {
                    (
                        p["image1_point"][0],
                        p["image1_point"][1],
                        p["image2_point"][0],
                        p["image2_point"][1],
                    )
                    for p in data["point_pairs"]
                }

                for point in existing_points:
                    point_tuple = (
                        point["image1_point"][0],
                        point["image1_point"][1],
                        point["image2_point"][0],
                        point["image2_point"][1],
                    )
                    if point_tuple not in new_points_set:
                        data["point_pairs"].append(point)

        # Save to JSON file
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=4)

    def run(self):
        """Run the point mapper interface."""
        plt.show()


def calculate_homography(points_file):
    """
    Calculate homography matrix from point pairs.

    Args:
        points_file: Path to the JSON file with point pairs

    Returns:
        The homography matrix from image 1 to image 2
    """
    # Load the point pairs
    with open(points_file, "r") as f:
        data = json.load(f)

    if len(data["point_pairs"]) < 4:
        print("Need at least 4 point pairs to calculate homography.")
        return None

    # Extract point coordinates
    src_points = np.array(
        [p["image1_point"] for p in data["point_pairs"]], dtype=np.float32
    )
    dst_points = np.array(
        [p["image2_point"] for p in data["point_pairs"]], dtype=np.float32
    )

    # Calculate the homography matrix
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    print(
        f"Calculated homography matrix with {np.sum(status)} inliers out of {len(status)} points."
    )
    print("Homography matrix:")
    print(H)

    return H


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Map corresponding points between two images."
    )
    parser.add_argument("--image1", help="Path to the first image")
    parser.add_argument("--image2", help="Path to the second image")
    parser.add_argument(
        "--output",
        "-o",
        default="point_pairs.json",
        help="Output file for point pairs (default: point_pairs.json)",
    )
    parser.add_argument(
        "--calculate",
        "-c",
        action="store_true",
        help="Calculate homography from saved points file",
    )

    args = parser.parse_args()

    if args.calculate:
        # Calculate homography from existing points file
        calculate_homography(args.output)
    else:
        # Run the point mapper interface
        mapper = PointMapper(args.image1, args.image2, args.output)
        mapper.run()


if __name__ == "__main__":
    main()
