import numpy as np
import os
import open3d as o3d

# Загрузка данных калибровки
calib_dir = 'calib'
calib_file = os.path.join(calib_dir, '0000.txt')
calib_data = {}
with open(calib_file, 'r') as f:
    for line in f:
        key, value = line.split(': ')
        calib_data[key] = np.array([float(x) for x in value.split()])

# Загрузка данных лидара
lidar_dir = 'velodyne/0000'
lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
lidar_data = []
for file in lidar_files:
    lidar_file = os.path.join(lidar_dir, file)
    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    lidar_data.append(lidar_points)

# Загрузка данных изображений
image_dir = 'image_02/0000'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
image_data = [o3d.io.read_image(os.path.join(image_dir, f)) for f in image_files]

# Загрузка данных аннотаций
label_dir = 'labels_02'
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
label_data = []
for file in label_files:
    label_file = os.path.join(label_dir, file)
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append([float(x) for x in line.strip().split()])
    label_data.append(labels)

# Загрузка данных OxTS
oxts_dir = 'oxts'
oxts_files = sorted([f for f in os.listdir(oxts_dir) if f.endswith('.txt')])
oxts_data = []
for file in oxts_files:
    oxts_file = os.path.join(oxts_dir, file)
    oxts_data.append(np.loadtxt(oxts_file))

# Синхронизация и сопоставление данных
for i in range(len(lidar_data)):
    lidar_point_cloud = o3d.geometry.PointCloud()
    lidar_point_cloud.points = o3d.utility.Vector3dVector(lidar_data[i][:, :3])
    
    # Применение матриц трансформации для сопоставления данных
    lidar_point_cloud.transform(calib_data['Tr_velo_cam'])
    
    image = image_data[i]
    labels = label_data[i]
    oxts = oxts_data[i]
    
    # Предварительная обработка данных лидара
    voxel_size = 0.1
    lidar_point_cloud = lidar_point_cloud.voxel_down_sample(voxel_size)
    plane_model, inliers = lidar_point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    ground_points = lidar_point_cloud.select_by_index(inliers)
    object_points = lidar_point_cloud.select_by_index(inliers, invert=True)
    
    # Нормализация данных
    object_points.translate(-np.mean(object_points.points, axis=0))
    object_points.scale(1.0 / np.max(np.abs(object_points.points)), center=(0, 0, 0))
    
    # Сохранение предварительно обработанных данных
    o3d.io.write_point_cloud(f'preprocessed_lidar_{i}.ply', object_points)
    np.save(f'preprocessed_labels_{i}.npy', labels)
    o3d.io.write_image(f'preprocessed_image_{i}.png', image)
    np.save(f'preprocessed_oxts_{i}.npy', oxts)
