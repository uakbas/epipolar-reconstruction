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
- [Scene 1](point_clouds/scene_1.ply)
- [Scene 2](point_clouds/scene_2.ply)
- [Scene 3](point_clouds/scene_3.ply)
- [Scene 4](point_clouds/scene_4.ply)
- [Scene 5](point_clouds/scene_5.ply)### Local Viewing Options

You can also download the PLY files and view them locally using:
- [MeshLab](https://www.meshlab.net/) - Feature-rich 3D mesh viewer and editor
- [CloudCompare](https://www.cloudcompare.org/) - Point cloud processing software
- [Blender](https://www.blender.org/) - Professional 3D creation suite

## Setup & Usage

### View Online (Recommended)

1. Enable GitHub Pages in your repository settings:
   - Go to: Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main`, Folder: `/documents`
   - Click Save
2. Wait a few minutes for deployment
3. Visit: `https://uakbas.github.io/epipolar-reconstruction/`

### Run Viewer Locally

1. Clone this repository
2. Open `documents/index.html` in your browser
3. The viewer will load PLY files from the `point_clouds/` directory

### Requirements (for development)

```bash
pip install open3d numpy opencv-python
```

## Project Structure

```
epipolar-reconstruction/
├── output/              # Generated output files
├── point_clouds/        # Point cloud data (PLY files)
│   ├── scene_1.ply
│   ├── scene_2.ply
│   └── ...
├── documents/           # GitHub Pages website
│   └── index.html       # Interactive Three.js viewer
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
