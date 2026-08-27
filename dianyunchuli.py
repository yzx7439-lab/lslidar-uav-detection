"""
LiDAR 点云处理器
功能：背景建模、前景检测、目标跟踪、输出目标航迹

处理流程：
1. 背景建模 - 基于初始 N 帧建立静态背景体素模型
2. 前景提取 - 通过体素差异和距离阈值检测前景点
3. 聚类分割 - DBSCAN 聚类将前景点分组
4. 目标跟踪 - Kalman 滤波 + 匹配关联实现目标连续跟踪
5. 结果输出 - 封装为目标航迹数据通过队列发送
"""

import datetime
import queue
import threading
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def preprocess_pcd(pcd, voxel_size=0.12):
    """点云下采样 - 使用体素网格过滤减少点数"""
    return pcd.voxel_down_sample(voxel_size)


def icp_register(source, target, voxel_size=0.12, max_iter=5):
    """ICP点云配准 - 将源点云对齐到目标点云，返回4x4变换矩阵"""
    if len(source.points) == 0 or len(target.points) == 0:
        return np.eye(4)  # 空点云返回单位矩阵

    threshold = voxel_size * 3.0  # ICP 匹配阈值，通常为体素大小的3倍
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return reg.transformation


def transform_pcd(pcd, transform):
    """对点云应用刚性变换（旋转+平移）"""
    out = o3d.geometry.PointCloud(pcd)
    out.transform(transform)
    return out


def points_to_voxel_keys(points, voxel_size):
    """将点云坐标转换为体素网格索引 (整数值)"""
    return np.floor(points / voxel_size).astype(np.int32)


def voxel_key_tuple(key):
    """将体素索引数组转换为元组 (用于哈希表)"""
    return int(key[0]), int(key[1]), int(key[2])


def dilate_voxel_keys(voxel_keys, radius=1):
    """体素膨胀 - 将体素向周围扩展radius范围，增加背景覆盖区域"""
    if radius <= 0:
        return set(map(voxel_key_tuple, voxel_keys))

    offsets = np.array([
        [dx, dy, dz]
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ])

    voxel_arr = np.array(list(map(voxel_key_tuple, voxel_keys)))
    expanded = voxel_arr[:, np.newaxis] + offsets[np.newaxis, :]
    return set(map(tuple, expanded.reshape(-1, 3)))


def build_background_model_from_pcds(bg_pcds, voxel_size=0.25, occ_ratio=0.9, dilate_radius=0):
    """
    从多帧点云构建静态背景体素模型

    算法原理：
    1. 对所有背景帧进行下采样和两两ICP配准对齐
    2. 将点云体素化，统计每个体素被覆盖的帧数
    3. 保留出现频率 >= occ_ratio 的体素作为背景
    4. 可选进行体素膨胀(dilate)以扩展背景区域
    """
    if not bg_pcds:
        raise ValueError("背景点云为空，无法构建背景模型。")

    bg_pre = [preprocess_pcd(p, voxel_size=voxel_size) for p in bg_pcds]  # 下采样
    ref = bg_pre[0]  # 以第一帧为参考，避免单帧越界
    counts = {}  # 体素被覆盖次数统计
    merged_points = []  # 合并所有帧的点
    frame_stats = []  # 每帧统计信息

    for idx, pcd in enumerate(bg_pre):
        if idx == 0:
            aligned = pcd  # 第一帧直接作为参考
        else:
            # 后续帧与参考帧进行ICP配准对齐
            transform = icp_register(pcd, ref, voxel_size=voxel_size, max_iter=5)
            aligned = transform_pcd(pcd, transform)

        points = np.asarray(aligned.points)
        if len(points) > 0:
            merged_points.append(points)
        voxel_keys = points_to_voxel_keys(points, voxel_size)  # 体素化
        unique_keys = set(map(voxel_key_tuple, voxel_keys))  # 去重
        frame_stats.append({
            "frame_index": idx,
            "downsampled_points": int(len(points)),  # 下采样后点数
            "unique_voxels": int(len(unique_keys)),  # 独有体素数
        })
        for key in unique_keys:
            counts[key] = counts.get(key, 0) + 1  # 统计每个体素被覆盖次数

    min_count = max(1, int(np.ceil(len(bg_pre) * occ_ratio)))  # 最小覆盖次数阈值
    bg_keys = {key for key, count in counts.items() if count >= min_count}  # 保留高覆盖体素
    bg_keys = dilate_voxel_keys(bg_keys, radius=dilate_radius)  # 可选膨胀
    overlap_stats = {
        "count_ge_1": sum(1 for count in counts.values() if count >= 1),
        "count_ge_5": sum(1 for count in counts.values() if count >= 5),
        "count_ge_10": sum(1 for count in counts.values() if count >= 10),
        "count_ge_15": sum(1 for count in counts.values() if count >= 15),
        "count_ge_min": sum(1 for count in counts.values() if count >= min_count),
    }
    if not bg_keys:
        raise ValueError(
            f"背景体素为空，无法完成背景建模。min_count={min_count}, "
            f"overlap_stats={overlap_stats}, frame_stats={frame_stats}"
        )

    merged_bg = o3d.geometry.PointCloud()
    if merged_points:
        merged_bg.points = o3d.utility.Vector3dVector(np.vstack(merged_points))  # 合并点云

    bg_keys_np = np.array(list(bg_keys), dtype=np.int32).reshape(-1, 3)
    return ref, bg_keys, bg_keys_np, merged_bg, frame_stats, overlap_stats, min_count


def build_foreground_pcd(points, mask):
    """根据mask从点云中提取前景点"""
    fg = o3d.geometry.PointCloud()
    if len(points) > 0 and np.any(mask):
        fg.points = o3d.utility.Vector3dVector(points[mask])
    return fg


def compute_foreground_mask(points, bg_hash_sorted, bg_kdtree, voxel_size, bg_dist_thresh):
    """
    通过体素差异和距离阈值检测前景点

    检测逻辑：
    1. 将当前帧点云体素化，查找命中背景体素的点
    2. 对于未命中背景的点，查询到最近背景点的距离
    3. 距离超过 bg_dist_thresh 的点标记为前景
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    voxel_keys = points_to_voxel_keys(points, voxel_size)  # 当前帧体素化
    voxel_hash = voxel_keys @ np.array([1000000, 1000, 1], dtype=np.int64)  # 3D哈希压缩
    indices = np.searchsorted(bg_hash_sorted, voxel_hash)  # 二分查找背景体素
    indices = np.clip(indices, 0, len(bg_hash_sorted) - 1)
    voxel_hits = bg_hash_sorted[indices] == voxel_hash  # 是否命中背景

    fg_indices = np.where(~voxel_hits)[0]  # 未命中背景的点索引
    dists = np.full(len(points), np.inf)
    if len(fg_indices) > 0:
        dists[fg_indices], _ = bg_kdtree.query(points[fg_indices], k=1)  # 到最近背景点的距离

    bg_dists = dists
    return (~voxel_hits) & (bg_dists > bg_dist_thresh)  # 未命中且距离超过阈值 = 前景


def subtract_background(frame_pcd, ref_pcd, bg_kdtree, bg_hash_sorted, voxel_size=0.25, do_register=False,
                        bg_dist_thresh=0.50):
    """对单帧点云进行背景滤除，返回(处理后点云, 前景点云, 前景mask)"""
    proc = preprocess_pcd(frame_pcd, voxel_size=voxel_size)  # 下采样

    if do_register:
        # 可选：与参考帧进行ICP配准对齐
        transform = icp_register(proc, ref_pcd, voxel_size=voxel_size, max_iter=5)
        proc = transform_pcd(proc, transform)

    points = np.asarray(proc.points)
    if len(points) == 0:
        return proc, o3d.geometry.PointCloud(), np.zeros(0, dtype=bool)

    mask = compute_foreground_mask(points, bg_hash_sorted, bg_kdtree, voxel_size, bg_dist_thresh)
    fg = build_foreground_pcd(points, mask)
    return proc, fg, mask


def cluster_foreground(fg_pcd, eps=0.55, min_points=5):
    """
    使用 DBSCAN 算法对前景点进行聚类分割

    DBSCAN 参数：
    - eps: 邻域半径，超过此距离的点视为不同簇
    - min_points: 形成簇所需的最少点数
    """
    if len(fg_pcd.points) == 0:
        return []

    labels = np.array(fg_pcd.cluster_dbscan(eps=eps, min_points=min_points))  # DBSCAN聚类
    if labels.size == 0 or labels.max() < 0:
        return []  # 无有效聚类

    points = np.asarray(fg_pcd.points)
    clusters = []
    for label in range(labels.max() + 1):
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        clusters.append(build_cluster_from_indices(points, idx))
    return clusters


def build_cluster_from_indices(points, indices):
    """根据点索引构建聚类，计算聚类的几何特征（中心、边界框、体积等）"""
    indices = np.asarray(indices, dtype=int)
    cluster_points = points[indices]
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    aabb = cluster_pcd.get_axis_aligned_bounding_box()  # 轴对齐边界框
    extent = np.asarray(aabb.get_extent(), dtype=float)
    nonzero_extent = extent[extent > 1e-3]  # 忽略过小维度
    min_nonzero = float(np.min(nonzero_extent)) if len(nonzero_extent) > 0 else 1e-3
    max_side = float(np.max(extent))
    volume = float(np.prod(extent))
    aspect_ratio = max_side / min_nonzero if min_nonzero > 0 else float("inf")
    return {
        "fg_indices": indices,  # 前景点索引
        "center": np.asarray(aabb.get_center()),  # 聚类中心
        "extent": extent,  # 边界框尺寸
        "num_points": len(indices),  # 点数
        "volume": volume,  # 体积
        "max_side": max_side,  # 最大边长
        "aspect_ratio": float(aspect_ratio),  # 长宽比
    }


def merge_close_clusters(clusters, fg_points, merge_dist=10.0):
    """
    合并距离接近的聚类 - 使用BFS将距离小于merge_dist的聚类合并
    解决DBSCAN可能将同一个目标分割成多个簇的问题
    """
    if not clusters:
        return []

    centers = np.array([cluster["center"] for cluster in clusters], dtype=float)
    visited = np.zeros(len(clusters), dtype=bool)
    merged = []

    for i in range(len(clusters)):
        if visited[i]:
            continue
        stack = [i]
        group = []
        visited[i] = True

        while stack:
            current = stack.pop()
            group.append(current)
            dists = np.linalg.norm(centers - centers[current], axis=1)
            neighbors = np.where(dists <= merge_dist)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        merged_indices = np.unique(np.concatenate([clusters[idx]["fg_indices"] for idx in group]))
        merged.append(build_cluster_from_indices(fg_points, merged_indices))

    return merged


def attach_background_distance(clusters, bg_kdtree):
    """为每个聚类计算到最近背景点的距离"""
    for cluster in clusters:
        center = cluster["center"]
        dist, _ = bg_kdtree.query(center.reshape(1, -1), k=1)
        cluster["bg_distance"] = float(dist[0])
    return clusters


def attach_frame_motion(clusters, prev_clusters, match_dist=2.0):
    """
    计算每个聚类与前一帧聚类的最近匹配距离
    用于判断目标是否有运动
    """
    if not prev_clusters:
        for cluster in clusters:
            cluster["frame_motion"] = 0.0  # 无前一帧数据
        return clusters

    if not clusters:
        return clusters

    prev_centers = np.array([cluster["center"] for cluster in prev_clusters], dtype=float)
    centers = np.array([cluster["center"] for cluster in clusters], dtype=float)
    dists_matrix = np.linalg.norm(centers[:, np.newaxis] - prev_centers[np.newaxis, :], axis=2)
    nearest = np.min(dists_matrix, axis=1)
    motions = np.where(nearest <= match_dist, nearest, match_dist)  # 截断过大的值

    for cluster, motion in zip(clusters, motions):
        cluster["frame_motion"] = float(motion)
    return clusters


def filter_candidates(clusters, config):
    """
    候选目标过滤 - 固定雷达无人机探测

    硬过滤条件：
    - 运动速度: 0-10 m/s (不排除悬停，大疆御3最大速度9m/s)
    - 目标距离: 10-300 m (大疆御3有效探测范围)
    - 聚类点数: >= 5 (少于5点不可信)
    - 最大边长: < 1.5 m (排除大型障碍物)
    - 长宽比: < 3 (排除极端细长目标)
    - 背景距离: > 10 m (10米内无背景点视为孤立目标)
    """
    out = []
    for cluster in clusters:
        extent = cluster["extent"]
        center = cluster["center"]
        center_range = np.linalg.norm(center)
        bg_distance = cluster.get("bg_distance", 0.0)
        frame_motion = cluster.get("frame_motion", 0.0)

        if cluster["num_points"] < config.min_points:
            continue
        if frame_motion < 0.0 or frame_motion > config.max_motion:
            continue
        if center_range < config.min_range or center_range > config.max_range:
            continue
        if bg_distance < config.min_bg_distance:
            continue
        if cluster.get("max_side", float("inf")) > config.max_side:
            continue
        if cluster.get("aspect_ratio", float("inf")) > config.max_aspect_ratio:
            continue

        cluster["range"] = center_range
        out.append(cluster)
    return out


def choose_best_candidate(candidates):
    """
    选择最佳候选 - 孤立度最高(bg_distance最大)的候选

    多候选时返回距离背景最远的目标（最孤立）
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.get("bg_distance", 0.0))


def match_candidate_to_prediction(candidates, predicted_pos, max_dist):
    """在锁定状态下按预测位置进行最近邻关联，避免多目标时跳变。"""
    if not candidates or predicted_pos is None:
        return None, None

    best = None
    best_dist = None
    for candidate in candidates:
        dist = float(np.linalg.norm(candidate["center"] - predicted_pos))
        if dist <= max_dist and (best_dist is None or dist < best_dist):
            best = candidate
            best_dist = dist
    return best, best_dist


def update_tentative_track(tentative, candidate, match_dist):
    """
    更新暂态跟踪状态

    暂态跟踪用于在确认锁定前验证目标稳定性：
    - 连续多帧检测到距离相近的目标
    - 累计运动距离达到阈值
    """
    if candidate is None:
        return None

    if tentative is None:
        return {
            "center": candidate["center"].copy(),
            "count": 1,  # 连续命中计数
            "motion": 0.0,  # 累计运动距离
            "candidate": candidate,
        }

    step_motion = np.linalg.norm(candidate["center"] - tentative["center"])
    if step_motion <= match_dist:  # 距离匹配，加入暂态跟踪
        return {
            "center": candidate["center"].copy(),
            "count": tentative["count"] + 1,
            "motion": tentative["motion"] + step_motion,
            "candidate": candidate,
        }

    # 距离不匹配，重新开始
    return {
        "center": candidate["center"].copy(),
        "count": 1,
        "motion": 0.0,
        "candidate": candidate,
    }


class KalmanFilter3D:
    """
    3D Kalman滤波器 - 用于目标状态估计和预测

    状态向量: [x, y, z, vx, vy, vz] (位置 + 速度)
    观测向量: [x, y, z] (仅位置)
    """

    def __init__(self, dt=1.0, process_var=0.1, meas_var=0.05):
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 10.0
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.Q = np.eye(6) * process_var
        self.R = np.eye(3) * meas_var
        self.initialized = False

    def init_state(self, pos):
        self.x[:3, 0] = pos.reshape(3)
        self.x[3:, 0] = 0.0
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3, 0].copy()

    def update(self, pos):
        z = pos.reshape(3, 1)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(6) - k @ self.H) @ self.P

    def position(self):
        return self.x[:3, 0].copy()

    def velocity(self):
        return self.x[3:, 0].copy()


def radar_to_enu(x, y, z, north_angle):
    """
    将雷达坐标系下的目标位置转换为地理东-北-天(ENU)坐标。

    已知：north_angle 表示雷达 y 轴相对正北顺时针角。
    假设雷达坐标系满足右手系且 z 轴向上。
    """
    theta = np.radians(north_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    dE = x * cos_theta + y * sin_theta
    dN = -x * sin_theta + y * cos_theta
    dU = z
    return dE, dN, dU


def calc_target_geolocation(x, y, z, lat0, lon0, alt0, north_angle):
    """
    计算目标经纬高（基于WGS84椭球体）

    参数:
        x: 目标雷达坐标系 x 坐标（米）
        y: 目标雷达坐标系 y 坐标（米）
        z: 目标雷达坐标系 z 坐标（米）
        lat0: 雷达纬度（度）
        lon0: 雷达经度（度）
        alt0: 雷达海拔高度（米）
        north_angle: 雷达 y 轴相对正北顺时针角（度）

    返回:
        (target_lat, target_lon, target_alt): 目标经纬高
    """
    dE, dN, dU = radar_to_enu(x, y, z, north_angle)
    horizontal = np.sqrt(dE**2 + dN**2)
    if horizontal < 0.001:
        return lat0, lon0, alt0 + dU

    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2

    lat_rad = np.radians(lat0)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    M = a * (1 - e2) / (1 - e2 * sin_lat**2)**1.5
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    m_per_deg_lat = M * np.pi / 180.0
    m_per_deg_lon = N * cos_lat * np.pi / 180.0

    dLat = dN / m_per_deg_lat
    dLon = dE / m_per_deg_lon

    return lat0 + dLat, lon0 + dLon, alt0 + dU


@dataclass
class ProcessorConfig:
    # ==================== 背景建模参数 ====================
    warmup_frames: int = 30  # 背景建模所需的帧数，收集足够帧后建立背景模型
    voxel: float = 0.12  # 体素大小(米)，用于下采样和背景体素化
    occ_ratio: float = 0.8  # 体素占用率阈值，表示该体素在80%帧中出现即判定为背景
    bg_dilate: int = 0  # 背景体素膨胀半径，0表示不膨胀

    # ==================== 聚类参数 ====================
    dbscan_eps: float = 0.8  # DBSCAN邻域半径(米)，超过此距离的点视为不同簇
    dbscan_min: int = 3  # DBSCAN最小聚类点数，少于此点数不形成簇
    merge_dist: float = 2.0  # 聚类合并距离阈值(米)，距离小于此值则合并两个簇

    # ==================== 目标跟踪参数 ====================
    max_match_dist: float = 6.0  # 预锁定阶段候选目标与跟踪目标的最大匹配距离(米)
    lock_match_dist: float = 4.0  # 已锁定阶段的最大匹配距离，比预锁定更严格
    max_missed: int = 5  # 目标丢失后最大允许的预测保持帧数，超过此值则判定锁定丢失
    confirm_frames: int = 2  # 确认锁定所需的连续检测帧数
    confirm_motion: float = 0.0  # 确认锁定所需的最小累计运动距离(米)
    dt: float = 0.2  # Kalman滤波器时间步长(秒)

    # ==================== 候选目标过滤参数 ====================
    min_points: int = 3           # 最小聚类点数，少于此值视为噪声
    max_motion: float = 10.0      # 最大帧间位移阈值(米/帧)，超出视为异常目标
    min_range: float = 10.0       # 最小探测距离 m
    max_range: float = 200.0     # 最大探测距离 m
    min_bg_distance: float = 0.2 # 到背景最小距离 m，小于该值通常视为贴近静态背景
    max_side: float = 2.5         # 边界框最大边长 m
    max_aspect_ratio: float = 18.0 # 最大长宽比，超过视为非目标
    bg_dist_thresh: float = 2.0  # 前景检测的距离阈值(米)，超过此距离背景点才标记为前景

    # ==================== 配准参数 ====================
    register_target: bool = False  # 是否启用帧间ICP配准，启用会增加计算开销但适合动态场景

    # ==================== 目标属性参数 ====================
    local_latitude: float = 0.0  # 雷达纬度(度)，部署时通过启动参数配置
    local_longitude: float = 0.0  # 雷达经度(度)，部署时通过启动参数配置
    local_altitude: float = 0.0  # 雷达海拔高度(米)，部署时通过启动参数配置
    north_angle: float = 0.0  # 雷达北向角(度)，部署时通过启动参数配置
    track_type: int = 1  # 航迹类别，用于输出数据标识目标类型
    track_batch: int = 1005  # 航迹批号，用于输出数据标识跟踪批次
    target_snr: int = 30  # 目标信噪比，用于输出数据


class LSPointCloudProcessor:
    """
    LiDAR 点云处理器主类

    处理流程：
    1. handle_frame() - 接收点云帧，触发背景建模或检测处理
    2. _build_background() - 从初始 N 帧建立背景模型
    3. _process_detection_frame() - 执行前景检测、聚类、跟踪
    4. _update_track() - Kalman 滤波 + 目标关联 + 状态管理
    5. _build_sender_record() - 封装目标航迹数据

    目标状态机：
    - 未锁定: 无候选目标，等待检测到目标
    - 预锁定: 连续 N 帧检测到同一目标，进入暂态跟踪
    - 已锁定: 目标跟踪稳定，输出航迹
    - 预测保持: 目标暂时丢失，基于速度预测保持跟踪
    - 锁定丢失: 连续 M 帧未检测到目标，放弃跟踪
    """

    def __init__(self, output_queue=None, config=None, on_background_ready=None, background_build_callback=None):
        self.output_queue = output_queue if output_queue is not None else queue.Queue()  # 输出队列
        self.config = config if config is not None else ProcessorConfig()  # 配置参数
        self.on_background_ready = on_background_ready  # 背景建模完成回调
        self.background_build_callback = background_build_callback  # 背景建模完成外部回调（用于清空队列）
        self.background_frames = []  # 背景帧缓存
        self.background_ready = False  # 背景是否就绪
        self.ref_pcd = None  # 参考点云
        self.bg_kdtree = None  # 背景KD树（用于最近邻查询）
        self.bg_hash_sorted = None  # 背景体素哈希（用于快速查表）
        self.prev_clusters = []  # 上一帧聚类结果
        self.kf = KalmanFilter3D(dt=self.config.dt)  # Kalman滤波器
        self.track_confirmed = False  # 是否已确认锁定
        self.tentative_track = None  # 暂态跟踪信息
        self.hit_streak = 0  # 连续命中计数
        self.miss_streak = 0  # 连续丢失计数
        self.predicted_pos = None  # Kalman预测位置
        self.frame_id = 0  # 帧计数
        self.lock = threading.Lock()  # 线程锁
        self.running = True  # 运行标志

    def handle_frame(self, points):
        """
        处理一帧点云数据

        工作流程：
        1. 如果背景未建模，收集帧用于建模
        2. 背景建模完成后，执行前景检测和目标跟踪
        """
        with self.lock:
            if not self.running:
                return None
            self.frame_id += 1
            xyz_i = np.asarray(points, dtype=np.float32)  # xyz + intensity
            if xyz_i.ndim != 2 or xyz_i.shape[1] < 3:
                raise ValueError("输入点云必须是形状为 (N, 3) 或 (N, 4+) 的二维数组。")
            point_count = len(xyz_i)
            print(f"[Processor] Received frame {self.frame_id:06d} with {point_count} points")

            if point_count == 0:
                print("[Processor] Empty frame skipped")
                return None

            frame_pcd = self._points_to_pcd(xyz_i[:, :3])  # 转为Open3D点云

            if not self.background_ready:
                self.background_frames.append(frame_pcd)  # 收集背景帧
                print(f"[Processor] Background warmup {len(self.background_frames)}/{self.config.warmup_frames}")
                if len(self.background_frames) >= self.config.warmup_frames:
                    try:
                        self._build_background()  # 建模
                    except Exception as exc:
                        print(f"[Processor] Background build failed: {exc}")
                        self.background_frames = []
                        self.background_ready = False
                return None

            return self._process_detection_frame(xyz_i, frame_pcd)

    def _build_background(self):
        """从收集的背景帧构建背景模型"""
        print(f"[Processor] Building background model from {len(self.background_frames)} frames...")
        ref_pcd, bg_keys, bg_keys_np, merged_bg, frame_stats, overlap_stats, min_count = \
            build_background_model_from_pcds(
                self.background_frames,
                voxel_size=self.config.voxel,
                occ_ratio=self.config.occ_ratio,
                dilate_radius=self.config.bg_dilate,
            )

        print(f"[Processor] Background stats | voxel={self.config.voxel} | occ_ratio={self.config.occ_ratio} | "
              f"min_count={min_count} | overlap={overlap_stats}")
        for stat in frame_stats:
            print(f"[Processor] Background frame {stat['frame_index'] + 1:02d} | "
                  f"downsampled_points={stat['downsampled_points']} | unique_voxels={stat['unique_voxels']}")

        if len(merged_bg.points) == 0:
            raise ValueError("背景融合结果为空，无法计算背景距离。")

        self.ref_pcd = ref_pcd
        self.bg_kdtree = cKDTree(np.asarray(merged_bg.points))  # 构建KD树加速最近邻查询
        bg_hash = bg_keys_np @ np.array([1000000, 1000, 1], dtype=np.int64)  # 体素哈希
        self.bg_hash_sorted = np.sort(bg_hash)
        self.background_ready = True
        self.background_frames = []
        print(f"[Processor] Background ready | voxels={len(bg_keys)} | merged_points={len(merged_bg.points)}")
        if self.on_background_ready is not None:
            self.on_background_ready()
        if self.background_build_callback is not None:
            print("[Processor] Calling background_build_callback")
            self.background_build_callback()

    def _process_detection_frame(self, xyz_i, frame_pcd):
        """
        处理检测帧：背景分割 -> 聚类 -> 过滤 -> 跟踪 -> 输出
        """
        # 背景分割：滤除属于静态背景的点
        proc, fg, fg_mask = subtract_background(
            frame_pcd, self.ref_pcd, self.bg_kdtree, self.bg_hash_sorted,
            voxel_size=self.config.voxel,
            do_register=self.config.register_target,
            bg_dist_thresh=self.config.bg_dist_thresh,
        )

        # 前景聚类：DBSCAN将前景点分组
        raw_clusters = cluster_foreground(fg, eps=self.config.dbscan_eps, min_points=self.config.dbscan_min)
        fg_points = np.asarray(fg.points)

        # 合并近距离聚类
        clusters = merge_close_clusters(raw_clusters, fg_points, merge_dist=self.config.merge_dist)

        # 计算每个聚类到背景的距离
        clusters = attach_background_distance(clusters, self.bg_kdtree)

        # 计算帧间运动
        clusters = attach_frame_motion(clusters, self.prev_clusters, match_dist=self.config.max_match_dist)

        # 过滤候选目标
        candidates = filter_candidates(clusters, self.config)

        fg_count = int(np.sum(fg_mask)) if len(fg_mask) > 0 else 0

        # 更新目标跟踪状态
        best, track_status, extrapolated = self._update_track(candidates)
        self.prev_clusters = clusters

        # 打印帧状态信息
        print(f"[Processor] Frame {self.frame_id:06d} | 前景点数={fg_count} | "
              f"原始聚类数量={len(raw_clusters)} | 合并后聚类数量={len(clusters)} | 候选目标数={len(candidates)}")
        if self.track_confirmed and self.predicted_pos is not None:
            pred_str = np.array2string(self.predicted_pos, precision=2, suppress_small=True)
            print(f"[Processor] Track status={track_status} | miss={self.miss_streak} | predicted={pred_str}")
        else:
            print(f"[Processor] Track status={track_status}")

        if best is None and self.predicted_pos is None:
            return None

        target_data = self._build_sender_record(best, xyz_i, proc, extrapolated)

        if best is not None:
            center = best["center"]
            dist = np.linalg.norm(center)
            print(
                f"[目标] X={center[0]:.3f} Y={center[1]:.3f} Z={center[2]:.3f} | 距离={dist:.1f}m | 高度={center[2]:.3f}m")
        elif self.predicted_pos is not None:
            pred = self.predicted_pos
            dist = np.linalg.norm(pred)
            print(
                f"[目标] X={pred[0]:.3f} Y={pred[1]:.3f} Z={pred[2]:.3f} | 距离={dist:.1f}m | 高度={pred[2]:.3f}m [外推]")

        return target_data

    def _update_track(self, candidates):
        """
        目标跟踪状态机

        状态转换：
        未锁定 -> (检测到候选) -> 预锁定 -> (连续命中+运动) -> 已锁定
        已锁定 -> (丢失候选) -> 预测保持 -> (超过max_missed) -> 锁定丢失 -> 未锁定
        """
        best = None
        extrapolated = False
        track_status = "未锁定"
        confirm_hits = max(1, self.config.confirm_frames)

        if not self.track_confirmed:
            # 预锁定阶段
            if candidates:
                seed = choose_best_candidate(candidates)
                self.tentative_track = update_tentative_track(self.tentative_track, seed, self.config.max_match_dist)
                if self.tentative_track is not None:
                    best = self.tentative_track["candidate"]
                    if not self.kf.initialized:
                        self.kf.init_state(best["center"])
                    self.kf.update(best["center"])
                    self.hit_streak = self.tentative_track["count"]
                    self.miss_streak = 0
                    track_status = f"预锁定({self.hit_streak})"
                    # 满足确认条件，进入已锁定
                    if self.tentative_track["count"] >= confirm_hits and \
                            self.tentative_track["motion"] >= self.config.confirm_motion:
                        self.kf.init_state(best["center"])
                        self.kf.update(best["center"])
                        self.track_confirmed = True
                        self.predicted_pos = self.kf.position()
                        track_status = "已锁定"
                        self.tentative_track = None
            else:
                self.tentative_track = None
                self.hit_streak = 0
                self.predicted_pos = None
                if not self.track_confirmed:
                    self.kf = KalmanFilter3D(dt=self.config.dt)
        else:
            # 已锁定阶段
            predicted = self.kf.predict()  # Kalman预测
            self.predicted_pos = predicted
            best, match_distance = match_candidate_to_prediction(
                candidates, predicted, self.config.lock_match_dist
            )
            if best is not None:
                self.kf.update(best["center"])  # 融合观测
                self.predicted_pos = self.kf.position()
                self.hit_streak += 1
                self.miss_streak = 0
                track_status = f"已锁定-更新(match={match_distance:.2f}m)"
            else:
                self.miss_streak += 1
                extrapolated = True
                track_status = f"预测保持({self.miss_streak})"
                if self.miss_streak > self.config.max_missed:
                    self._reset_track_state()
                    track_status = "锁定丢失"
                    extrapolated = False
        return best, track_status, extrapolated

    def _reset_track_state(self):
        """重置跟踪状态到初始值"""
        self.kf = KalmanFilter3D(dt=self.config.dt)
        self.track_confirmed = False
        self.tentative_track = None
        self.hit_streak = 0
        self.miss_streak = 0
        self.predicted_pos = None

    def _build_sender_record(self, best_cluster, xyz_i, proc_pcd, extrapolated):
        """
        封装目标航迹数据为发送格式

        包含：位置、速度、角度、时间戳、调试信息等
        """
        now = datetime.datetime.now()

        if best_cluster is not None:
            center = best_cluster["center"]
            fg_idx = best_cluster["fg_indices"]
            proc_xyz = np.asarray(proc_pcd.points)
            cluster_points = proc_xyz[fg_idx] if len(proc_xyz) > 0 else np.zeros((0, 3), dtype=np.float32)
            intensity = self._estimate_cluster_intensity(xyz_i, cluster_points)
        elif self.predicted_pos is not None:
            center = self.predicted_pos  # 使用预测位置
            intensity = 0
        else:
            return None

        x, y, z = [float(v) for v in center]
        east, north, up = radar_to_enu(x, y, z, self.config.north_angle)
        distance = float(np.linalg.norm(center))  # 球面距离
        horizontal = float(np.sqrt(east**2 + north**2))  # 地理水平距离
        azimuth = np.degrees(np.arctan2(east, north)) % 360.0  # 方位角，以北为0°顺时针
        elevation = np.degrees(np.arctan2(up, horizontal)) if horizontal > 1e-6 else (90.0 if up >= 0 else -90.0)
        velocity = float(np.linalg.norm(self.kf.velocity())) if self.kf.initialized else 0.0
        quality = 8 if self.track_confirmed else 5  # 已锁定质量8，预锁定5

        target_lat, target_lon, target_alt = calc_target_geolocation(
            x, y, z,
            self.config.local_latitude,
            self.config.local_longitude,
            self.config.local_altitude,
            self.config.north_angle
        )

        print(f"[目标] 经度={target_lon:.8f} 纬度={target_lat:.8f} 高度={target_alt:.3f}m | "
              f"方位={azimuth:.2f}° 俯仰={elevation:.2f}° 距离={distance:.1f}m")

        target = {
            "航迹类别": self.config.track_type,
            "批号": self.config.track_batch,
            "时": now.hour,
            "分": now.minute,
            "秒": now.second,
            "毫秒": now.microsecond // 1000,
            "目标纬度": round(float(target_lat), 8),
            "目标经度": round(float(target_lon), 8),
            "目标海拔高度": round(float(target_alt), 3),
            "目标距离": int(round(distance)),
            "目标方位": round(float(azimuth), 2),
            "目标俯仰": round(float(elevation), 2),
            "本地纬度": self.config.local_latitude,
            "本地经度": self.config.local_longitude,
            "本地海拔高度": self.config.local_altitude,
            "航速": int(round(max(0.0, velocity))),
            "航迹质量": quality,
            "目标强度": int(round(intensity)),
            "目标信噪比": self.config.target_snr,
            "外推标示": 1 if extrapolated else 0,
            "航向": round(float(azimuth), 2),
            "备份": 0x00,
            "调试中心X": round(x, 3),
            "调试中心Y": round(y, 3),
            "调试中心Z": round(z, 3),
            "调试速度": round(velocity, 3),
        }
        return target

    def _estimate_cluster_intensity(self, xyz_i, cluster_points):
        """估算聚类的平均强度值（使用最近邻查询）"""
        if len(cluster_points) == 0 or len(xyz_i) == 0 or xyz_i.shape[1] < 4:
            return 0.0

        raw_xyz = xyz_i[:, :3]
        raw_i = xyz_i[:, 3]  # 强度通道
        tree = cKDTree(raw_xyz)
        dists, idx = tree.query(cluster_points, k=1)
        valid = np.isfinite(dists)
        if not np.any(valid):
            return 0.0
        return float(np.mean(raw_i[idx[valid]]))

    @staticmethod
    def _points_to_pcd(points_xyz):
        """将numpy数组转换为Open3D点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
        return pcd

    def stop(self):
        """停止处理器，清理资源"""
        with self.lock:
            self.running = False
