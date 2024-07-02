import numpy as np
import cv2
import open3d as o3d

# Внутренняя калибровка камеры
def calibrate_camera(images, grid_size, square_size):
    # Находим углы шахматной доски на изображениях
    objpoints = []
    imgpoints = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        if ret:
            objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # Оценка внутренних параметров камеры
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# Внешняя калибровка камеры и лидара
def calibrate_extrinsic(lidar_points, image_points, camera_matrix):
    # Находим соответствия между 3D-точками лидара и 2D-точками на изображении
    lidar_pts = np.array([pt.xyz for pt in lidar_points])
    image_pts = np.array([pt.uv for pt in image_points])
    
    # Оценка экзогенных параметров
    _, rvec, tvec, _ = cv2.solvePnPRansac(lidar_pts, image_pts, camera_matrix, None)
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3, 1)
    
    # Матрица преобразования камера -> лидар
    extrinsic = np.hstack((R, T))
    return extrinsic

# Пример использования
images = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg'), ...]
grid_size = (9, 6)
square_size = 0.025  # размер клеток шахматной доски в метрах
camera_matrix, dist_coeffs = calibrate_camera(images, grid_size, square_size)

lidar_points = [o3d.geometry.PointCloud.create_from_file('lidar_data.pcd')]
image_points = [ImagePoint(u, v, x, y, z) for u, v, x, y, z in zip(uv_coords, lidar_pts)]
extrinsic = calibrate_extrinsic(lidar_points, image_points, camera_matrix)
