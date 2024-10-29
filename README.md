# CV-540
## Environment
Environment:
matplotlib: 3.9.2 
opencv-python: 4.10.0
open3d: 0.18.0 
torch: 1.13.1 
torchvision: 0.14.1 
CUDA version: 11.7 
Python 3.10.15

## Usage
```bash
# Generate depth map from 'inputs' to 'outputs' directory
python run.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 80 --img-path './inputs' --outdir './outputs'
# Save the depth map as a point cloud file
python depth_to_pointcloud.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 20 --img-path './inputs' --outdir './outputs'
# Generate a point cloud of the original view
python 3dpoints.py

## Refernces
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
