# Monocular estimation of pothole depth and filling of potholes  
![1](https://github.com/user-attachments/assets/91831463-53ce-44e8-b98f-491037be84b6)
## Environment  
matplotlib: 3.9.2  
opencv-python: 4.10.0  
open3d: 0.18.0  
torch: 1.13.1  
torchvision: 0.14.1  
CUDA version: 11.7  
Python 3.10.15  
scipy 1.13.1  

## Image Data Acquisition  
Images for this project were sourced from two datasets: [Road Pothole Detection Dataset2024](https://aistudio.baidu.com/datasetdetail/292614) and [Pothole Dataset](https://public.roboflow.com/object-detection/pothole).  

The pothole pavement images used were subjected to target detection using the [YOLO11](https://github.com/ultralytics/ultralytics) model, which yields the pothole roadway images used later to generate depth maps and point clouds.

## Usage
###
Download the weights file like [depth_anything_v2_metric_vkitti_vitl.pth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main) and put it in the checkpoints directory.  

### Generate depth map from 'inputs' to 'outputs' directory
```bash
python run.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 80 --img-path './inputs' --outdir './outputs'
```

### Save the depth map as a point cloud file
```bash
python depth_to_pointcloud.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 20 --img-path './inputs' --outdir './outputs'
```

### Generate a point cloud of the original view
```bash
python 3dpoints.py
```

### Pothole filling of individual point clouds by fitting planes  
```bash
python fill_plane.py ----file "Path_to_Your_Point_Cloud_file"
```

### Pothole filling of individual point clouds by fitting Curved surface  
```bash
python fill_mesh.py ----file "Path_to_Your_Point_Cloud_file"
```

### Pothole filling of individual point clouds by curvature  
```bash
python fill_curvature.py ----file "Path_to_Your_Point_Cloud_file"
```

### Using the QT interface to show code  
```bash
python qt.py
```


## Stament  
This project is an assignment for CV540.  

## Refernces
The URL of the original article is [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main)  

```bash
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```
