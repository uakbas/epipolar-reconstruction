# Epipolar Reconstruction

A 3D reconstruction system using epipolar geometry and stereo vision to reconstruct fish objects from multi-view orthogonal camera setups.

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

**Output:**
- High-quality 3D point clouds in PLY format
- Multi-view masks for each scene
- Interactive web-based visualization

## 3D Reconstruction Results

View the reconstructed fish objects as interactive 3D point clouds!

### 🎨 Interactive 3D Viewer (GitHub Pages)

**[View All Scenes in Interactive Viewer](https://uakbas.github.io/epipolar-reconstruction/)** ✨

The web-based viewer features:
- 🔄 Rotate, zoom, and pan with mouse controls
- 📊 Real-time point cloud rendering with Three.js
- 🎯 Adjustable point size slider
- 📱 Responsive design
- 🌐 No installation required - runs in your browser!

> **What is GitHub Pages?** GitHub Pages is a free static website hosting service that turns your repository into a live website. Once enabled, your HTML files are accessible at `https://username.github.io/repository-name/`. The viewer loads your PLY files dynamically and renders them in 3D.

> **Note:** PLY files are not directly viewable on the GitHub repository page itself (GitHub doesn't have a built-in PLY viewer). However, the GitHub Pages viewer provides full interactive 3D visualization!

### 📦 Reconstruction Results

Each scene contains:
- 3D point cloud (PLY format)
- Camera masks from 4 orthogonal views (front, back, top, bottom)

**Download Complete Scenes:**
- [Scene 1](results/scene_1/) - [Point Cloud](results/scene_1/scene_1.ply) | Masks: [Front](results/scene_1/front.png), [Back](results/scene_1/back.png), [Top](results/scene_1/top.png), [Bottom](results/scene_1/bottom.png)
- [Scene 2](results/scene_2/) - [Point Cloud](results/scene_2/scene_2.ply) | Masks: [Front](results/scene_2/front.png), [Back](results/scene_2/back.png), [Top](results/scene_2/top.png), [Bottom](results/scene_2/bottom.png)
- [Scene 3](results/scene_3/) - [Point Cloud](results/scene_3/scene_3.ply) | Masks: [Front](results/scene_3/front.png), [Back](results/scene_3/back.png), [Top](results/scene_3/top.png), [Bottom](results/scene_3/bottom.png)
- [Scene 4](results/scene_4/) - [Point Cloud](results/scene_4/scene_4.ply) | Masks: [Front](results/scene_4/front.png), [Back](results/scene_4/back.png), [Top](results/scene_4/top.png), [Bottom](results/scene_4/bottom.png)
- [Scene 5](results/scene_5/) - [Point Cloud](results/scene_5/scene_5.ply) | Masks: [Front](results/scene_5/front.png), [Back](results/scene_5/back.png), [Top](results/scene_5/top.png), [Bottom](results/scene_5/bottom.png)

### 📚 Theoretical Resources

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

### 🛠️ Local Viewing Options

Download the PLY files and view them locally using:
- [MeshLab](https://www.meshlab.net/) - Feature-rich 3D mesh viewer and editor
- [CloudCompare](https://www.cloudcompare.org/) - Point cloud processing software
- [Blender](https://www.blender.org/) - Professional 3D creation suite

## Project Structure

```
epipolar-reconstruction/
├── results/             # Reconstruction results (PLY + masks per scene)
│   ├── scene_1/
│   │   ├── scene_1.ply  # 3D point cloud
│   │   ├── front.png    # Camera mask (front view)
│   │   ├── back.png     # Camera mask (back view)
│   │   ├── top.png      # Camera mask (top view)
│   │   └── bottom.png   # Camera mask (bottom view)
│   ├── scene_2/
│   └── ...
├── docs/                # GitHub Pages website (interactive viewer)
│   ├── index.html       # Three.js point cloud viewer
│   └── point_clouds/    # PLY files for web viewer
├── theory_resources/    # Theoretical papers and documentation (PDFs)
├── depth/               # Depth estimation module
└── scenes/              # Original scene data with images and masks
```

## Technologies

- **Point Cloud Reconstruction**: Epipolar geometry and stereo vision
- **Web Viewer**: Three.js with PLYLoader
- **Hosting**: GitHub Pages (free static site hosting)

## License

MIT License

## License

MIT License
