# Image Stitching & Panorama Construction

A computer vision project implementing two related techniques from scratch: **background stitching** (merging images with moving foregrounds) and **panorama construction** (multi-image seamless stitching) — using keypoint detection, homography, and OpenCV.

## Two Modes

### 1. Background Stitching
Merges two images that share the same background but have different moving foreground objects (e.g., people walking through a scene).

**Steps:**
1. Extract keypoints from both images
2. Extract and match features across keypoints
3. Detect overlapping regions between image pairs
4. Compute homography matrix between overlapping pairs
5. Transform and stitch — eliminating moving foreground without cropping

### 2. Image Panorama
Replicates the panorama feature of modern cameras by stitching multiple sequential photos into one seamless wide image.

**Steps:**
1. Extract features from each image (SIFT/ORB)
2. Match features across adjacent pairs
3. Compute homography for alignment
4. Warp and blend images into a single mosaic

## Tech Stack

`Python` · `OpenCV` · `NumPy` · `SIFT / ORB` · `Homography`

## Setup

```bash
git clone https://github.com/mitalildeshpande/Background-Stitching-and-Image-Panorama.git
cd Background-Stitching-and-Image-Panorama
pip install opencv-python numpy
python stitch.py
```