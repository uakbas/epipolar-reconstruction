# Epipolar Reconstruction

A 3D reconstruction system using epipolar geometry and stereo vision to reconstruct fish objects from multi-view orthogonal camera setups.
Results: https://uakbas.github.io/epipolar-reconstruction/


## Overview

This project implements a complete pipeline for 3D point cloud reconstruction using epipolar geometry principles. The system is specifically designed for fish object reconstruction with the following characteristics:

**Camera Setup:**
- **Orthogonal Configuration**: Cameras are positioned perpendicular to each other around the target object
- **Multi-View**: Typically uses 4 cameras (can use fewer), capturing views from different angles (front, back, top, bottom)
- **Fixed Positions**: Cameras maintain fixed positions to enable accurate geometric reconstruction

**Input Requirements:**
- **Masks Required**: Binary masks are essential for each camera view to isolate the target object from the background
- **Target Object**: Optimized for fish object reconstruction, though adaptable to other objects
- **Calibrated Views**: Orthogonal camera positions enable simplified epipolar geometry calculations


## 🎨 3D Reconstruction Results [ Interactive 3D Viewer | GitHub Pages ]
View the reconstructed fish objects as interactive 3D point clouds:
https://uakbas.github.io/epipolar-reconstruction/


## 📚 Theoretical Resources

PDF documents covering the theoretical foundations:
- [Epipolar Geometry](theory_resources/03-epipolar-geometry.pdf)
- [12.1 Epipolar Geometry](theory_resources/12.1_Epipolar_Geometry.pdf)
- [12.2 Essential Matrix](theory_resources/12.2_Essential_Matrix.pdf)
- [12.3 Fundamental Matrix](theory_resources/12.3_Fundamental_Matrix.pdf)
- [12.4 8-Point Algorithm](theory_resources/12.4_8Point_Algorithm.pdf)
- [13.1 Stereo Rectification](theory_resources/13.1_Stereo_Rectification.pdf)
- [14.4 Alignment Lucas-Kanade](theory_resources/14.4_Alignment__LucasKanade.pdf)
- [14.5 Alignment Baker-Matthews](theory_resources/14.5_Alignment__BakerMatthews.pdf)
- [15.1 Tracking KLT](theory_resources/15.1_Tracking__KLT.pdf)
- [15 RANSAC Notes](theory_resources/15-RANSAC-notes.pdf)
- [370 RANSAC](theory_resources/370_10_RANSAC.pptx.pdf)
- [Feature Matching](theory_resources/lecture_4_2_feature_matching.pdf)
- [8-Point Algorithm (Original Paper)](theory_resources/8%20point%20algorithm%20-%20original%20paper.pdf)
- [Structure-from-Motion Revisited](theory_resources/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)

## License

MIT License
