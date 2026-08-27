# LSLiDAR UAV Detection

基于镭神 LSS4 激光雷达原始点云的低空无人机检测与单目标跟踪原型。项目直接接收 UDP 数据包，完成点云解析、静态背景差分、候选目标聚类、卡尔曼跟踪和地理坐标计算。

> 本项目是独立的研究与工程原型，不包含厂商 SDK、上位机、驱动程序、手册或采集数据，也不代表镭神智能官方实现。

## 处理流程

```text
LSS4 UDP 数据
  -> 帧同步与协议解析
  -> 极坐标转 XYZ 点云
  -> 多帧静态背景体素建模
  -> 体素差分 + KD-Tree 前景提取
  -> DBSCAN 聚类与几何特征过滤
  -> 三维 Kalman 滤波与状态机跟踪
  -> ENU/WGS84 坐标转换
  -> 航迹字段输出
```

## 主要能力

- 监听 UDP `2368` 端口，按照 LSS4 单回波数据格式完成帧同步和向量化解析。
- 使用体素降采样、多帧 ICP 配准和占用率统计构建静态背景模型。
- 结合体素哈希与 `cKDTree` 最近邻距离提取动态前景。
- 使用 DBSCAN、AABB 尺寸、点数、长宽比、距离及帧间位移筛选无人机候选点簇。
- 使用三维匀速 Kalman 滤波器和“预锁定—已锁定—预测保持—锁定丢失”状态机维持连续航迹。
- 根据雷达位置和北向角计算目标经纬度、海拔、方位、俯仰、距离和航速。
- 使用进程、线程和消息队列解耦数据接收、点云处理和结果消费。

## 环境要求

- Windows 或 Linux
- Python 3.10–3.12
- 支持 UDP 数据输出的 LSS4 激光雷达

安装依赖：

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 使用方法

确保本机网卡与雷达处于同一网段，并将雷达数据端口配置为本机 UDP `2368`。在项目目录运行：

```bash
python run_pipeline.py \
  --port 2368 \
  --latitude 0.0 \
  --longitude 0.0 \
  --altitude 0.0 \
  --north-angle 0.0
```

请将示例中的位置和北向角替换为实际标定值。`north-angle` 表示雷达 y 轴相对正北的顺时针角度；可以运行 `python radar_north_angle.py`，通过已知无人机位置辅助计算。

当前主流程以控制台调试为主，检测结果已封装为 Python 字典，但尚未实现正式网络协议发送。如需接入其他系统，可在 `run_pipeline.py` 的结果消费循环中增加序列化或发送逻辑。

## 效果演示

- [无人机点云检测演示](demo_videos/uav_detection_demo.mp4)
- [无人机连续跟踪演示](demo_videos/uav_tracking_demo.mp4)

视频为实际处理效果录屏。GitHub 网页端如不能直接播放，可点击链接下载后查看。

## 核心文件

- `LSlistener.py`：UDP 接收、帧同步和原始点云解析。
- `dianyunchuli.py`：背景建模、前景提取、聚类、跟踪及航迹封装。
- `run_pipeline.py`：实时处理流水线和进程生命周期管理。
- `radar_north_angle.py`：雷达北向角辅助标定工具。

## 参数调优

检测参数集中在 `ProcessorConfig`。部署前应根据点云密度、场景距离和背景复杂度重点调整：

- `voxel`、`occ_ratio`：背景模型的空间分辨率和稳定体素比例。
- `bg_dist_thresh`：前景点到背景的最小距离。
- `dbscan_eps`、`dbscan_min`：点簇邻域半径和最少点数。
- `min_range`、`max_range`、`max_side`、`max_aspect_ratio`：候选目标几何门限。
- `lock_match_dist`、`max_missed`：航迹关联门限和短时漏检保持时间。

## 已知限制

- 当前实现面向固定雷达和相对稳定的城市背景，移动平台需要额外的位姿补偿。
- 当前只维护一条主要目标航迹，不是多目标跟踪器。
- 候选识别依赖人工配置的几何与运动阈值，未使用深度学习分类模型。
- 仓库不提供采集数据，因此需要使用真实雷达数据进行完整运行验证。

## License

本项目采用 [MIT License](LICENSE)。第三方软硬件及其文档遵循各自的许可证和使用条款。
