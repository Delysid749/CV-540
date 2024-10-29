import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("outputs/image5.ply")
# 创建旋转矩阵（沿 x 轴旋转 π弧度即180°）
R = pcd.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]))
# 应用旋转
pcd.rotate(R, center=(0, 0, 0))


#使用 RANSAC 检测平面
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"平面方程: {a}x + {b}y + {c}z + {d} = 0")

# 计算旋转矩阵，使得平面法向量与 Z 轴对齐
normal_vector = np.array([a, b, c])
target_axis = np.array([0, 1, 0])
# 计算旋转轴和旋转角度
v = np.cross(normal_vector, target_axis)  # 旋转轴
s = np.linalg.norm(v)                # 旋转轴的模
c = np.dot(normal_vector, target_axis)    # 旋转角的余弦值

# 构建旋转矩阵（使用罗德里格斯公式）
vx = np.array([
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0]
])

R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

#应用旋转矩阵
pcd.rotate(R, center=(0, 0, 0))



# 显示点云
o3d.visualization.draw_geometries([pcd])