#!/usr/bin/env python3
"""
verify_fundamental.py: Interactive point mapping for three images A, B, C; compute fundamentals A↔B and C↔B via DLT;
visualize epipolar lines with opacity and overlay both sets on image B for intersection check.

Usage:
    python verify_fundamental.py A.jpg B.jpg C.jpg
"""
import argparse
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PointMapper:
    def __init__(self, img1_path, img2_path, output_path=None):
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        if self.img1 is None or self.img2 is None:
            raise ValueError("Could not load one or both images.")
        self.img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.output_path = output_path or "point_pairs.json"
        self.point_pairs = []
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                if data.get('image1') == os.path.abspath(self.img1_path) and \
                   data.get('image2') == os.path.abspath(self.img2_path):
                    for p in data.get('point_pairs', []):
                        x1, y1 = p['image1_point']
                        x2, y2 = p['image2_point']
                        self.point_pairs.append([x1, y1, x2, y2])
            except Exception:
                pass
        self.selecting_img1 = True
        self.current = [None, None]
        self.setup()

    def setup(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax1.imshow(self.img1_rgb);
        self.ax1.set_title(os.path.basename(self.img1_path)); self.ax1.axis('off')
        self.ax2.imshow(self.img2_rgb);
        self.ax2.set_title(os.path.basename(self.img2_path)); self.ax2.axis('off')
        self.fig.suptitle(f"Click points: {os.path.basename(self.img1_path)} → {os.path.basename(self.img2_path)}. Enter=save, Esc=exit")
        self.instruction = self.fig.text(0.5, 0.01,
            f"Select in Image1 | Count: {len(self.point_pairs)}", ha='center')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()

    def on_click(self, event):
        if event.inaxes is None: return
        x, y = event.xdata, event.ydata
        if self.selecting_img1 and event.inaxes == self.ax1:
            self.current[0] = [x, y]
            self.selecting_img1 = False
            self.ax1.plot(x, y, 'ro'); self.fig.canvas.draw()
        elif not self.selecting_img1 and event.inaxes == self.ax2:
            self.current[1] = [x, y]
            self.point_pairs.append([*self.current[0], *self.current[1]])
            idx = len(self.point_pairs)
            c = plt.cm.tab10(idx % 10)
            self.ax1.plot(*self.current[0], 'o', color=c)
            self.ax2.plot(x, y, 'o', color=c)
            self.ax1.text(self.current[0][0]+5, self.current[0][1]+5, str(idx), color=c)
            self.ax2.text(x+5, y+5, str(idx), color=c)
            self.selecting_img1 = True
            self.current = [None, None]
            self.instruction.set_text(f"Select in Image1 | Count: {idx}")
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter':
            self.save(); print(f"Saved {len(self.point_pairs)} to {self.output_path}")
        elif event.key == 'escape':
            plt.close(self.fig)

    def redraw(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.imshow(self.img1_rgb); self.ax1.axis('off')
        self.ax2.imshow(self.img2_rgb); self.ax2.axis('off')
        for i, (x1, y1, x2, y2) in enumerate(self.point_pairs, 1):
            c = plt.cm.tab10(i % 10)
            self.ax1.plot(x1, y1, 'o', color=c)
            self.ax2.plot(x2, y2, 'o', color=c)
            self.ax1.text(x1+5, y1+5, str(i), color=c)
            self.ax2.text(x2+5, y2+5, str(i), color=c)
        self.instruction.set_text(f"Select in {'Image1' if self.selecting_img1 else 'Image2'} | Count: {len(self.point_pairs)}")
        self.fig.canvas.draw()

    def save(self):
        data = { 'image1': os.path.abspath(self.img1_path), 'image2': os.path.abspath(self.img2_path), 'point_pairs': [] }
        for x1,y1,x2,y2 in self.point_pairs:
            data['point_pairs'].append({'image1_point':[int(x1),int(y1)], 'image2_point':[int(x2),int(y2)]})
        with open(self.output_path,'w') as f: json.dump(data, f, indent=4)

    def run(self): plt.show()


def normalize_points(pts):
    m = pts.mean(axis=0); c = pts - m
    d = np.linalg.norm(c,axis=1).mean()
    s = np.sqrt(2)/d if d>0 else 1.0
    T = np.array([[s,0,-s*m[0]],[0,s,-s*m[1]],[0,0,1]])
    pts_h = np.hstack([pts,np.ones((len(pts),1))])
    return (T@pts_h.T).T[:,:2], T


def eight_point_dlt(pts1,pts2):
    if len(pts1)<8: raise ValueError
    p1n,T1 = normalize_points(pts1)
    p2n,T2 = normalize_points(pts2)
    A = np.array([[x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1,1]
                  for (x1,y1),(x2,y2) in zip(p1n,p2n)])
    _,_,Vt = np.linalg.svd(A)
    F0=Vt[-1].reshape(3,3)
    U,S,Vt2 = np.linalg.svd(F0); S[2]=0
    F = T2.T@(U@np.diag(S)@Vt2)@T1
    return F/F[2,2] if abs(F[2,2])>1e-8 else F


def overlay_epilines(img, pts_src, F, which, color, alpha=0.6):
    # which=1 or 2 indicates image index for cv2.computeCorrespondEpilines
    overlay = img.copy()
    lines = cv2.computeCorrespondEpilines(pts_src.reshape(-1,1,2), which, F).reshape(-1,3)
    for (a,b,c) in lines:
        h,w = img.shape[:2]
        if abs(b)>1e-6:
            p0=(0,int(-c/b)); p1=(w,int(-(c+a*w)/b))
        else:
            x0=int(-c/a); p0,p1=(x0,0),(x0,h)
        cv2.line(overlay,p0,p1,color,2)
    return cv2.addWeighted(overlay,alpha,img,1-alpha,0)


def load_corr(fn):
    """Load correspondences JSON and return two Nx2 arrays."""
    with open(fn, 'r') as f:
        data = json.load(f)
    pts1 = np.array([p['image1_point'] for p in data['point_pairs']], dtype=float)
    pts2 = np.array([p['image2_point'] for p in data['point_pairs']], dtype=float)
    return pts1, pts2


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('A'); parser.add_argument('B'); parser.add_argument('C'); args=parser.parse_args()
    # A<->B
    print('--- A↔B mapping ---')
    PointMapper(args.A,args.B,'pts_AB.json').run()
    ptsA,ptsB1 = load_corr('pts_AB.json')
    F_AB = eight_point_dlt(ptsA,ptsB1)
    print('F_AB:',np.round(F_AB,6))
    # C<->B
    print('--- C↔B mapping ---')
    PointMapper(args.C,args.B,'pts_CB.json').run()
    ptsC,ptsB2 = load_corr('pts_CB.json')
    F_CB = eight_point_dlt(ptsC,ptsB2)
    print('F_CB:',np.round(F_CB,6))
    # overlay both on B
    imgB=cv2.imread(args.B)
    # AB epilines in green
    imgB1=overlay_epilines(imgB,ptsA,F_AB,2,(0,255,0),alpha=0.7)
    # CB epilines in blue
    imgB2=overlay_epilines(imgB1,ptsC,F_CB,2,(255,0,0),alpha=0.7)
    cv2.imwrite('combined_B_epilines.png',imgB2)
    print('Saved combined epilines on B as combined_B_epilines.png')

if __name__=='__main__': main()
