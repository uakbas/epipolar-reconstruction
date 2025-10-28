# Epipolar Reconstruction

3D reconstruction project using epipolar geometry and stereo vision.

## 3D Reconstruction Results

This project generates 3D point clouds from stereo image pairs. View the interactive results below!

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

### 📦 Download Point Clouds

Original PLY files:
- [Scene 1](docs/point_clouds/scene_1.ply)
- [Scene 2](docs/point_clouds/scene_2.ply)
- [Scene 3](docs/point_clouds/scene_3.ply)
- [Scene 4](docs/point_clouds/scene_4.ply)
- [Scene 5](docs/point_clouds/scene_5.ply)

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
- [Structure-from-Motion Revisited](theory_resources/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)### Local Viewing Options

You can also download the PLY files and view them locally using:
- [MeshLab](https://www.meshlab.net/) - Feature-rich 3D mesh viewer and editor
- [CloudCompare](https://www.cloudcompare.org/) - Point cloud processing software
- [Blender](https://www.blender.org/) - Professional 3D creation suite

## Setup & Usage

### View Online (Recommended)

1. Enable GitHub Pages in your repository settings:
   - Go to: Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main`, Folder: `/docs` ← **Use /docs (GitHub Pages requirement)**
   - Click Save
2. Wait a few minutes for deployment
3. Visit: `https://uakbas.github.io/epipolar-reconstruction/`

### Run Viewer Locally

1. Clone this repository
2. Open `docs/index.html` in your browser
3. The viewer will load PLY files from the `docs/point_clouds/` directory

### Requirements (for development)

```bash
pip install open3d numpy opencv-python
```

## Project Structure

```
epipolar-reconstruction/
├── output/              # Generated output files
├── docs/                # GitHub Pages website (viewer + data)
│   ├── index.html       # Interactive Three.js viewer
│   └── point_clouds/    # Point cloud data (PLY files)
│       ├── scene_1.ply
│       ├── scene_2.ply
│       └── ...
├── theory_resources/    # Theoretical papers and documentation (PDFs)
│   ├── 03-epipolar-geometry.pdf
│   ├── 12.1_Epipolar_Geometry.pdf
│   └── ...
├── depth/               # Depth estimation module
├── scene_objects/       # Scene object data
└── README.md
```

## Technologies

- **Point Cloud Reconstruction**: Epipolar geometry and stereo vision
- **Web Viewer**: Three.js with PLYLoader
- **Hosting**: GitHub Pages (free static site hosting)

## License

MIT License
