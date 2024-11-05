import argparse
import copy
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree

# adjust view to XOY plane and show points cloud
def draw_geometries(geometries, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Set view to XOY plane
    ctr = vis.get_view_control()
    all_points = np.vstack([np.asarray(pcd.points) for pcd in geometries])
    centroid = np.mean(all_points, axis=0)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(centroid)
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()
    return 0

# This function rotates the point cloud pcd to be parallel to the XOZ plane.
def rotate(point_cloud):
  # 计算最小外接矩形
  obb = point_cloud.get_oriented_bounding_box()

  # 获取外接矩形的旋转矩阵和中心点
  center = obb.center
  R = obb.R

  # 将点云旋转至XOZ平面
  # 计算旋转角度
  angle_x = np.arctan2(R[1, 2], R[2, 2])  # 绕X轴旋转
  angle_y = np.arctan2(-R[0, 2], np.sqrt(R[0, 0]**2 + R[0, 1]**2))  # 绕Y轴旋转
  angle_z = np.arctan2(R[0, 1], R[0, 0])  # 绕Z轴旋转

  # 创建旋转矩阵
  R_x = np.array([[1, 0, 0],
                  [0, np.cos(angle_x), -np.sin(angle_x)],
                  [0, np.sin(angle_x), np.cos(angle_x)]])

  R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                  [0, 1, 0],
                  [-np.sin(angle_y), 0, np.cos(angle_y)]])

  R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                  [np.sin(angle_z), np.cos(angle_z), 0],
                  [0, 0, 1]])

  # 组合旋转矩阵
  R_combined = R_z @ R_y @ R_x

  # 将点云旋转到XOZ平面
  points = np.asarray(point_cloud.points)
  rotated_points = (R_combined @ points.T).T

  # Apply an additional 90° rotation around the X-axis to make the image_plane perpendicular to the XOY image_plane
  R_90_x = np.array([[1, 0, 0],
                     [0, 0, -1],
                     [0, 1, 0]])
  # Apply an additional 180° rotation around the Y-axis to image_rotate counterclockwise in the XOZ image_plane
  R_180_y = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])

  final_rotated_points = (R_90_x @ rotated_points.T).T
  final_rotated_points = (R_180_y @ final_rotated_points.T).T
  # 更新原有点云的坐标
  point_cloud.points = o3d.utility.Vector3dVector(final_rotated_points)
  return point_cloud

# define a function to fit plane by Least Squares Method
def generate_plane(point_cloud):
    # Extract points from the point cloud
    points = np.asarray(point_cloud.points)

    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(points - centroid, rowvar=False)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, 0]

    # Compute the plane offset (D) using the centroid
    D = -normal_vector.dot(centroid)

    # Compute the minimum bounding box of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    density = 100000
    # Generate random points within the bounding box
    random_points = np.random.rand(density, 3) * (max_bound - min_bound) + min_bound
    # Project the random points onto the plane
    distances = (random_points @ normal_vector + D) / np.linalg.norm(normal_vector)
    plane_points = random_points - np.outer(distances, normal_vector)


    # Filter the plane points to be within the bounding box of the original point cloud
    within_bounds = np.all((plane_points >= min_bound) & (plane_points <= max_bound), axis=1)
    plane_points = plane_points[within_bounds]

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

    # Apply Moving Least Squares (MLS) smoothing
    plane_pcd = plane_pcd.voxel_down_sample(voxel_size=0.01)
    plane_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Set the color of the plane to gray
    plane_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return plane_pcd

# The function is coloured and divided into inner and outer points based on the distance between point clouds
def colorize_distance_and_segment(image_rotate, image_plane):

    # Extract the y-coordinates of the points in 'image_rotate'
    rotate_y = np.asarray(image_rotate.points)[:, 1]

    # Extract the y-coordinates of the points in 'image_plane'
    plane_y = np.asarray(image_plane.points)[:, 1]

    # Build a KDTree for the plane points
    plane_tree = KDTree(plane_y[:, np.newaxis])
    # Query the KDTree for the nearest neighbor distances( Shortest straight line between points and surfaces )
    point_y_distances, _ = plane_tree.query(rotate_y[:, np.newaxis])

    # Normalize the distances to the range 0-255
    point_y_distances_normalized = 255 * (point_y_distances - point_y_distances.min()) / (point_y_distances.max() - point_y_distances.min())

    for i in range(len(rotate_y)):
        if rotate_y[i] < plane_y.min():
            point_y_distances_normalized[i] = 0

    # Set the colors based on the normalized distances
    colors = np.zeros((rotate_y.shape[0], 3))
    colors[:, 0] = point_y_distances_normalized  # Set the red channel based on the distances
    colors[:, 1] = point_y_distances_normalized  # Set the green channel based on the distances
    colors[:, 2] = point_y_distances_normalized  # Set the blue channel based on the distances

    # Apply the colors to the point cloud
    rotate_clone = copy.deepcopy(image_rotate)
    rotate_clone.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Set the threshold for inlines (30% of 255)
    threshold = 0.3 * 255

    # Split the points into inlines and outliers
    inlines_indices = np.where(point_y_distances_normalized > threshold)[0]
    outliers_indices = np.where(point_y_distances_normalized <= threshold)[0]

    # Create inlines and outlier point clouds
    inlines = image_rotate.select_by_index(inlines_indices)
    outliers = image_rotate.select_by_index(outliers_indices)
    return inlines, outliers,rotate_clone

# This function fills the points according to the distance of the point cloud from its upper boundary (the fitted road surface))
def filled_pothole(pcd):
    # 读取点云
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # 获取颜色信息
    # 计算点云的边界框
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 确定平面的中心和尺寸
    weight = max_bound[0] - min_bound[0]  # 箱子X轴方向的长度
    height = max_bound[2] - min_bound[2]  # 箱子Z轴方向的宽度
    # 设置填充点的间距，例如为该距离的1%
    y_distance = (max_bound[1]-min_bound[1])/100
    depth = max_bound[1]-min_bound[1]
    # 用于存储填充后的点
    filled_points = []
    # 遍历原始点云中的每个点，检查是否需要填充
    for point in points:
        # 生成该点到上截面之间的填充点
        y_levels = np.arange(min_bound[1], point[1], y_distance)
        for y in y_levels:
            filled_points.append([point[0], y, point[2]])

    # 将填充后的点转换为点云格式
    filled_points = np.array(filled_points)
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(filled_points)

    # 计算坑洼区域的平均颜色
    average_color = np.mean(colors, axis=0)
    filled_pcd.paint_uniform_color(average_color)

    return filled_pcd, depth

# This function uses slicing to calculate the volume of the filled point cloud.
def compute_volume_slicing(pcd, slice_thickness=0.01):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    volume = 0.0

    # 沿Y轴切片
    y_slices = np.arange(min_bound[1], max_bound[1], slice_thickness)
    for y in y_slices:
        slice_points = points[(points[:, 1] >= y) & (points[:, 1] < y + slice_thickness)]
        if len(slice_points) >= 3:
            hull = ConvexHull(slice_points[:, [0, 2]])
            area = hull.volume  # 使用凸包的体积作为切片的面积
            volume += area * slice_thickness

    return volume

def main():
    parser = argparse.ArgumentParser(description="Process point cloud data.")
    parser.add_argument("--file", type=str, required=True, help="Path to the point cloud file")
    args = parser.parse_args()

    # Load Point Cloud
    point_cloud = o3d.io.read_point_cloud(args.file)
    draw_geometries([point_cloud], window_name="Oriented Bounding Box")

    # Visualisation of the rotated point cloud
    pcd_rotate = rotate(point_cloud)
    draw_geometries([pcd_rotate], window_name="Rotated Point Cloud to XOZ Plane")

    # fit plane by Least Squares Method
    plane = generate_plane(pcd_rotate)
    draw_geometries([plane, pcd_rotate], window_name="Fitted Plane")

    # Delineation of inner and outer points
    inlines, outliers, colorize = colorize_distance_and_segment(pcd_rotate, plane)
    draw_geometries([colorize], window_name="Colored Point Cloud")
    draw_geometries([inlines], window_name="Inliers")

    # fill the potholes
    pcd_fill, depth = filled_pothole(inlines)
    draw_geometries([pcd_fill], window_name="Filled block")
    draw_geometries([pcd_fill, outliers], window_name="Filled Pothole")

    # Output depth and volume
    print(f"Estimated depth: {depth} m")
    volume1 = compute_volume_slicing(pcd_fill)
    print(f"Estimated volume (slicing): {volume1} m³")