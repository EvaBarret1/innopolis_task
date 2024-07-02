for i in range(len(X_RGB)):
    min_dist = inf
    closest_j = None
    for j in range(len(X_lidar)):
        dist = distance(X_i^RGB.bbox_center, X_j^lidar.bbox_center)
        if dist < min_dist:
            min_dist = dist
            closest_j = j
    if min_dist < distance_threshold:
        X_fused = {
            'bbox': 0.5 * (X_i^RGB.bbox + X_j^lidar.bbox),
            'score': max(X_i^RGB.score, X_j^lidar.score),
            'class': X_i^RGB.class_id,  # или можно выбрать класс с большей оценкой
            # другие объединенные данные
        }
        fused_detections.append(X_fused)
    else:
        fused_detections.append(X_i^RGB)
        fused_detections.append(X_j^lidar)
