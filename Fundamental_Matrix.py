#!/usr/bin/env python3
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
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    existing = json.load(f)
                    if existing.get("image1") == os.path.abspath(self.img1_path) and \
                       existing.get("image2") == os.path.abspath(self.img2_path):
                        for pair in existing.get("point_pairs", []):
                            x1, y1 = pair["image1_point"]
                            x2, y2 = pair["image2_point"]
                            self.point_pairs.append([x1, y1, x2, y2])
            except (json.JSONDecodeError, IOError):
                pass

        self.selecting_img1 = True
        self.current_pair = [None, None]
        self.setup_display()

    def setup_display(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax1.imshow(self.img1_rgb)
        self.ax1.set_title(f"Image 1: {os.path.basename(self.img1_path)}")
        self.ax1.axis("off")

        self.ax2.imshow(self.img2_rgb)
        self.ax2.set_title(f"Image 2: {os.path.basename(self.img2_path)}")
        self.ax2.axis("off")

        self.fig.suptitle(
            "Click corresponding points: first in Image 1, then in Image 2.\n"
            "Enter=save, Esc=exit, Backspace=undo"
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.instruction_text = self.fig.text(
            0.5, 0.01,
            f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
            f"Pairs: {len(self.point_pairs)}",
            ha="center"
        )
        self.redraw()

    def on_click(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        if self.selecting_img1 and event.inaxes == self.ax1:
            self.current_pair[0] = [x, y]
            self.selecting_img1 = False
            self.ax1.plot(x, y, 'ro')
            self.fig.canvas.draw()
        elif not self.selecting_img1 and event.inaxes == self.ax2:
            self.current_pair[1] = [x, y]
            self.point_pairs.append([
                self.current_pair[0][0], self.current_pair[0][1],
                self.current_pair[1][0], self.current_pair[1][1]
            ])
            idx = len(self.point_pairs)
            color = plt.cm.tab10(idx % 10)
            self.ax1.plot(self.current_pair[0][0], self.current_pair[0][1], 'o', color=color)
            self.ax2.plot(x, y, 'o', color=color)
            self.ax1.text(self.current_pair[0][0]+5, self.current_pair[0][1]+5, str(idx), color=color)
            self.ax2.text(x+5, y+5, str(idx), color=color)
            self.selecting_img1 = True
            self.current_pair = [None, None]
            self.instruction_text.set_text(
                f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
                f"Pairs: {len(self.point_pairs)}"
            )
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter':
            self.save_points()
            print(f"Saved {len(self.point_pairs)} pairs to {self.output_path}")
        elif event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 'backspace':
            if self.point_pairs:
                self.point_pairs.pop()
                self.redraw()

    def redraw(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.imshow(self.img1_rgb); self.ax1.axis('off')
        self.ax2.imshow(self.img2_rgb); self.ax2.axis('off')
        for i, (x1, y1, x2, y2) in enumerate(self.point_pairs, start=1):
            color = plt.cm.tab10(i % 10)
            self.ax1.plot(x1, y1, 'o', color=color)
            self.ax2.plot(x2, y2, 'o', color=color)
            self.ax1.text(x1+5, y1+5, str(i), color=color)
            self.ax2.text(x2+5, y2+5, str(i), color=color)
        self.instruction_text.set_text(
            f"Select a point in {'Image 1' if self.selecting_img1 else 'Image 2'} | "
            f"Pairs: {len(self.point_pairs)}"
        )
        self.fig.canvas.draw()

    def save_points(self):
        data = {
            'image1': os.path.abspath(self.img1_path),
            'image2': os.path.abspath(self.img2_path),
            'point_pairs': []
        }
        for x1, y1, x2, y2 in self.point_pairs:
            data['point_pairs'].append({
                'image1_point': [int(x1), int(y1)],
                'image2_point': [int(x2), int(y2)]
            })
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        plt.show()


def compute_fundamental_from_points(points_file):
    """
    Compute the fundamental matrix from saved correspondences in a JSON file.
    """
    with open(points_file, 'r') as f:
        data = json.load(f)
    pts1 = np.array([p['image1_point'] for p in data['point_pairs']], dtype=np.float32)
    pts2 = np.array([p['image2_point'] for p in data['point_pairs']], dtype=np.float32)
    if pts1.shape[0] < 8:
        print("Error: At least 8 correspondences required.")
        return None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)
    if F is None:
        print("Fundamental matrix estimation failed.")
        return None
    inliers = int(mask.sum())
    print("Fundamental matrix F:")
    print(F)
    print(f"Inliers: {inliers} / {pts1.shape[0]}")
    return F


def main():
    parser = argparse.ArgumentParser(
        description="Select correspondences or compute fundamental matrix."
    )
    parser.add_argument("--image1", help="Path to the first image")
    parser.add_argument("--image2", help="Path to the second image")
    parser.add_argument(
        "--output", "-o", default="point_pairs.json",
        help="JSON file for point pairs (default: point_pairs.json)"
    )
    parser.add_argument(
        "--fundamental", "-f", action="store_true",
        help="Compute fundamental matrix from saved points"
    )
    args = parser.parse_args()

    if args.fundamental:
        compute_fundamental_from_points(args.output)
    else:
        if not args.image1 or not args.image2:
            parser.error("--image1 and --image2 are required when not computing fundamental.")
        mapper = PointMapper(args.image1, args.image2, args.output)
        mapper.run()


if __name__ == "__main__":
    main()


# python point_mapper_fundamental.py --image1 imgA.jpg --image2 imgB.jpg
# python point_mapper_fundamental.py --fundamental

