# **Computer Vision Assignments**  
This repository contains two assignments focused on advanced computer vision techniques, including geometric shape detection and 3D scene reconstruction.  

---

## ***Assignment 1: Hough Triangles***  
This assignment involves implementing detectors for the following types of triangles using the Hough Transform:  
- **Equilateral Triangles** – All sides and angles are equal.  
- **Isosceles Triangles** – Two sides of equal length with equal base angles.  
- **Right Triangles** – One angle is exactly 90 degrees.  

### Methodology:  
- Detection is based on processing the standard Hough Transform map used for line detection.  
- A sliding-window HT variant is applied to detect triangles of varying sizes.  
- Parameters (edge detection thresholds, HT resolution) are tuned per image.  
- Detected lines are color-coded:  
  - **Blue** for equilateral triangles.  
  - **Green** for isosceles triangles.  
  - **Red** for right triangles.  
 
---

## ***Assignment 2: Plane Detection from Stereo Images***  
This assignment focuses on identifying planar regions from a pair of images taken from different poses using the following steps:  
1. **Interest Point Detection:** Identifying key points in both images.  
2. **Feature Matching:** Finding potential matches and filtering outliers.  
3. **Matrix Estimation and Filtering:**  
   - Estimating the **Essential Matrix (E)** and computing the **Fundamental Matrix (F)**.  
   - Filtering matches inconsistent with E.  
4. **3D Reconstruction:**  
   - Extracting camera matrices and performing **Triangulation** to obtain a 3D point cloud.  
5. **Plane Fitting:**  
   - Sequential plane fitting using **RANSAC** by:  
     - Finding the best fitting plane.  
     - Removing supporting points.  
     - Repeating the process to identify multiple planes.  
6. **Visualization:**  
   - Displaying the normals of the detected planes. 
