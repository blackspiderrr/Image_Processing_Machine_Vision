# Image Processing and Machine Vision Lab & Project

This repository contains laboratory exercises and the final project for EIE3527 **Image Processing and Machine Vision** course at Tongji University.

## 🧪 Labs

Practical experiments focusing on computer vision tools (OpenCV), mathematical libraries (Eigen), and geometric algorithms.

* **Lab 1: Environment Setup**

Configuration of the C++ development environment with OpenCV and necessary dependencies.

* **Lab 2: OpenCV Basics**

Fundamental image processing operations using the OpenCV library.

* **Lab 3: Eigen Basics**

Introduction to the **Eigen** C++ template library for linear algebra, matrices, and vector operations.

* **Lab 4: Camera Geometry with Eigen**

Implementation of geometric transformations and camera models using Eigen matrix operations.

* **Lab 5: Fusion**

Techniques for data or sensor fusion (e.g., combining multi-source data).

* **Lab 6: Corner Detection & Circle Fitting**

Feature extraction tasks including detecting corners in images and fitting geometric shapes (circles) to data points.

---

## 🚀 Final Project

### Real-time Depth Map and 3D Point Cloud Generation based on Stereo Camera System

**Overview**
A complete C++/OpenCV system capable of real-time 3D reconstruction using a **HBVCAM-W202011HD V33** stereo camera. The system captures stereo images, calculates depth, and renders a dense 3D point cloud.

**Key Features & Implementation**

1. **Camera Calibration**:
* Used chessboard patterns to solve for intrinsics, extrinsics, and distortion coefficients.

* Achieved high-precision calibration with a mean reprojection error of **0.57 pixels**.

2. **Stereo Rectification & Matching**:
* **SIFT Feature Matching**: Employed SIFT descriptors with RANSAC and epipolar constraints to verify alignment.

* **Disparity Estimation**: Implemented and compared **Block Matching (BM)** vs. **Semi-Global Block Matching (SGBM)**. SGBM was optimized to reduce noise and improve edge preservation.

3. **3D Reconstruction**:
* **Depth Calculation**: Converted disparity maps to depth maps using triangulation based on the camera baseline (~66mm).

* **Point Cloud Visualization**: Generated a dense point cloud  and visualized it interactively using the **Pangolin** library.

**Performance**

* The system operates in real-time, successfully reconstructing complex scenes (pedestrians, objects).

* Real-world measurement accuracy (e.g., measuring object length) achieved an error rate of **< 10%**.
