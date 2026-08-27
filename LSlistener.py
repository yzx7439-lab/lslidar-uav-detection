"""
LSS4 LiDAR 帧数据接收器
功能：接收 UDP 数据包，进行帧分割，向量化解析点云数据，输出XYZ坐标
"""

import socket
import argparse
import numpy as np

# ============================================================================
# 常量定义
# ============================================================================

DATA_PORT = 2368                     # UDP 数据接收端口
PACKET_SIZE = 1206                  # 单个数据包大小（字节）

# 帧同步标识：标志一帧数据的结束和新帧的开始
FRAME_SYNC = bytes([0xff, 0xaa, 0xbb, 0xcc, 0xdd])

# 单回波模式参数
POINT_SIZE = 8                      # 每个点占用 8 字节
DATA_SIZE = 1192                    # 有效数据区 1192 字节


# ============================================================================
# LiDAR 点云解析器类（向量化高性能解析）
# ============================================================================

class LidarParser:
    def __init__(self):
        pass
    
    def parse_frame(self, data):
        """
        高性能向量化解析
        data: 原始帧数据字节
        返回: (n, 3) float32 numpy array，XYZ 坐标
        """
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 8)

        h_angle = ((raw[:, 0].astype(np.int32) << 8) | raw[:, 1].astype(np.int32))
        h_angle = np.where(h_angle > 32767, h_angle - 65536, h_angle) * 0.01
        
        temp_angle = raw[:, 2]
        sign = (temp_angle >> 4) & 0x01
        
        v_angle_raw = ((raw[:, 2].astype(np.int32) << 8) | raw[:, 3].astype(np.int32))
        v_angle = np.where(sign == 1, v_angle_raw | 0xE000, v_angle_raw)
        v_angle = np.where(v_angle > 32767, v_angle - 65536, v_angle) * 0.01
        
        distance = (raw[:, 4].astype(np.int32) << 16 |
                    raw[:, 5].astype(np.int32) << 8 |
                    raw[:, 6].astype(np.int32)) * 0.001
        
        valid = distance > 0
        h_angle = h_angle[valid]
        v_angle = v_angle[valid]
        distance = distance[valid]
        
        cos_v = np.cos(np.radians(v_angle))
        sin_v = np.sin(np.radians(v_angle))
        cos_h = np.cos(np.radians(h_angle))
        sin_h = np.sin(np.radians(h_angle))
        
        xyz = np.column_stack([
            distance * cos_v * sin_h,
            distance * cos_v * cos_h,
            distance * sin_v
        ])
        
        return xyz.astype(np.float32)


# ============================================================================
# LiDAR 帧数据接收器类
# ============================================================================

class LidarFrameReceiver:
    """
    LiDAR 帧数据接收器

    功能说明：
    1. 通过 UDP 接收 LiDAR 原始数据包
    2. 通过帧同步标识进行帧分割
    3. 向量化解析点云数据为笛卡尔坐标系
    """

    def __init__(self, port: int = DATA_PORT, output_queue=None):
        """
        初始化接收器
        :param port: UDP 监听端口
        :param output_queue: multiprocessing.Queue，用于发送 xyz 点云数据
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Windows 推荐设置
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)  # 8MB 起步

        self.sock.settimeout(1.0)
        self.sock.bind(('0.0.0.0', port))

        self.running = False
        self.current_frame_packets = []

        self.frame_count = 0
        self.total_packets = 0

        self.synced = False
        self.skip_first_frame = True
        self.total_points_detected = 0
        self.parser = LidarParser()
        self.output_queue = output_queue

    def run(self):
        """
        启动接收器主循环
        持续接收 UDP 数据包并处理，直到被中断或出错
        """
        self.running = True
        print(f"[接收器] 监听端口 {DATA_PORT}")
        print("=" * 70)

        while self.running:
            try:
                data, addr = self.sock.recvfrom(PACKET_SIZE)
                self.total_packets += 1
                self._process_packet(data)
            except socket.timeout:
                continue
            except OSError as e:
                if hasattr(e, 'winerror') and e.winerror == 10035:
                    continue
                print(f"[错误] {e}")
                continue
            except KeyboardInterrupt:
                print("\n[接收器] 用户中断")
                break
            except Exception as e:
                print(f"[错误] {e}")
                continue

        self._cleanup()

    def _process_packet(self, data: bytes):
        """
        处理单个 UDP 数据包
        :param data: 原始数据包字节数据
        """
        if len(data) < 4:
            return

        if len(data) != PACKET_SIZE:
            print(f"[警告] 数据包大小异常: {len(data)} 字节，预期 {PACKET_SIZE} 字节")
            return

        if not self.synced:
            if data.find(FRAME_SYNC) != -1:
                self.synced = True
                self.current_frame_packets = []
                self.skip_first_frame = True
                print(f"[同步] 帧同步完成 @ 数据包 #{self.total_packets}")

        if not self.synced:
            return

        self.current_frame_packets.append(data)

        frame_sync_found = False
        for i in range(0, DATA_SIZE - POINT_SIZE + 1, POINT_SIZE):
            if data[i:i + 5] == FRAME_SYNC:
                frame_sync_found = True
                break

        if frame_sync_found:
            self._save_frame()

    def _save_frame(self):
        """
        保存当前帧并输出调试信息
        """
        if not self.current_frame_packets:
            return

        if self.skip_first_frame:
            self.skip_first_frame = False
            self.current_frame_packets = []
            return

        self.frame_count += 1
        raw_frame = b''.join(p[:DATA_SIZE] for p in self.current_frame_packets)

        xyz = self.parser.parse_frame(raw_frame)

        if self.output_queue is not None:
            try:
                self.output_queue.put(xyz, timeout=0.1)
            except Exception:
                pass

        self.total_points_detected += len(xyz)
        self.current_frame_packets = []

    def _cleanup(self):
        """
        清理资源：关闭套接字，输出统计摘要
        """
        if self.sock:
            self.sock.close()
        self.running = False
        print("")
        print("=" * 70)
        print("[接收器] 已停止")
        print(f"  总帧数:      {self.frame_count}")
        print(f"  总数据包数:  {self.total_packets}")
        print(f"  总点云数:    {self.total_points_detected}")
        print("=" * 70)

    def stop(self):
        """停止接收器"""
        self.running = False


# ============================================================================
# 主函数入口
# ============================================================================

def main():
    """主函数入口"""
    parser = argparse.ArgumentParser(
        description="LSS4 LiDAR 帧数据接收器 - 仅帧分割和调试输出"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=DATA_PORT,
        help=f"UDP 端口 (默认: {DATA_PORT})"
    )
    args = parser.parse_args()

    receiver = LidarFrameReceiver(port=args.port)

    try:
        receiver.run()
    except KeyboardInterrupt:
        print("\n[接收器] 键盘中断")
        receiver.stop()


if __name__ == "__main__":
    main()