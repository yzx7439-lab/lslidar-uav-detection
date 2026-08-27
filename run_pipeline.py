"""
LiDAR 点云处理 pipeline 启动脚本
====================================

架构说明：
┌──────────────────────────────────────────────────────────────────┐
│  子进程: LidarFrameReceiver (LSlistener)                          │
│  - 接收 UDP 数据包                                                │
│  - 进行帧分割和解析                                               │
│  - 输出 xyz 点云数据到 input_queue                                │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │ input_queue (maxsize=10)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  主进程线程: feeder_worker                                         │
│  - 从 input_queue 取出 xyz 点云                                   │
│  - 调用 processor.handle_frame(xyz) 进行处理                       │
│  - 将处理结果放入 output_queue                                    │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │ output_queue
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  主线程: 结果消费循环                                              │
│  - 从 output_queue 取出处理结果                                   │
│  - 目前仅打印输出 (debug 模式)                                     │
│  - 预留接口: 可扩展为保存文件/发送网络/转发等                        │
└──────────────────────────────────────────────────────────────────┘

用途：接收 UDP LiDAR 数据，进行点云处理，输出目标检测结果
"""

import argparse
import multiprocessing
import signal
import sys
import threading

from LSlistener import LidarFrameReceiver
from dianyunchuli import LSPointCloudProcessor, ProcessorConfig


def feeder_worker(input_queue, processor, output_queue, running_flag):
    """
    Feeder 循环：从 input_queue 取 xyz 点云数据，送给 processor 处理
    处理结果放入 output_queue

    参数:
        input_queue: multiprocessing.Queue, 接收来自 LSlistener 的 xyz 点云
        processor: LSPointCloudProcessor, 点云处理器实例
        output_queue: multiprocessing.Queue, 发送处理结果
        running_flag: multiprocessing.Array, 控制线程运行状态的标志
    """
    print("[Feeder] Started")
    while running_flag[0]:
        try:
            xyz = input_queue.get(timeout=0.5)  # 从 LiDAR 接收队列取点云
            result = processor.handle_frame(xyz)  # 送入处理器进行检测跟踪
            if result is not None:
                output_queue.put(result, timeout=0.1)  # 将检测结果放入输出队列
        except multiprocessing.queues.Empty:
            continue  # 队列为空，继续等待
        except Exception as e:
            print(f"[Feeder] Error: {e}")
            continue
    print("[Feeder] Stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="LSS4 无人机点云检测与跟踪")
    parser.add_argument("--port", type=int, default=2368, help="LiDAR UDP 监听端口")
    parser.add_argument("--latitude", type=float, default=0.0, help="雷达纬度（度）")
    parser.add_argument("--longitude", type=float, default=0.0, help="雷达经度（度）")
    parser.add_argument("--altitude", type=float, default=0.0, help="雷达海拔高度（米）")
    parser.add_argument(
        "--north-angle",
        type=float,
        default=0.0,
        help="雷达 y 轴相对正北的顺时针角度",
    )
    parser.add_argument("--track-batch", type=int, default=1005, help="输出航迹批号")
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建进程间队列
    # input_queue: LSlistener -> Feeder, 传输原始点云 xyz
    # output_queue: Processor -> 主线程, 传输检测结果
    input_queue = multiprocessing.Queue(maxsize=10)
    output_queue = multiprocessing.Queue()

    # 运行控制标志 (使用 Array 实现跨进程共享)
    running_flag = multiprocessing.Array('i', [1])  # 控制 feeder 线程
    output_running = multiprocessing.Array('i', [1])  # 控制主线程消费循环

    # 创建点云处理器实例
    config = ProcessorConfig(
        local_latitude=args.latitude,
        local_longitude=args.longitude,
        local_altitude=args.altitude,
        north_angle=args.north_angle,
        track_batch=args.track_batch,
    )
    processor = LSPointCloudProcessor(output_queue=output_queue, config=config)

    # 创建并启动 LidarFrameReceiver 子进程
    receiver = LidarFrameReceiver(port=args.port, output_queue=input_queue)
    receiver_process = multiprocessing.Process(target=receiver.run)
    receiver_process.start()

    # 启动 feeder 线程 (daemon 模式，随主进程退出而自动结束)
    feeder_thread = threading.Thread(
        target=feeder_worker,
        args=(input_queue, processor, output_queue, running_flag),
        daemon=True
    )
    feeder_thread.start()

    def shutdown():
        """
        优雅关闭函数：
        1. 设置停止标志
        2. 等待子进程和线程有序退出
        3. 清理资源
        """
        print("\n[Main] Shutting down...")
        running_flag[0] = 0  # 通知 feeder 线程停止
        output_running[0] = 0  # 通知主循环停止

        receiver.stop()  # 通知接收器停止
        receiver_process.join(timeout=2)  # 等待子进程退出
        if receiver_process.is_alive():
            receiver_process.terminate()  # 超时后强制终止

        # 清理队列资源
        try:
            input_queue.close()
            input_queue.join_thread()
        except Exception:
            pass

        processor.stop()  # 停止处理器
        print("[Main] Done")
        sys.exit(0)

    # 注册 Ctrl+C 信号处理
    signal.signal(signal.SIGINT, lambda s, f: shutdown())

    print("[Main] Pipeline started")
    print(f"  - LiDAR receiver: subprocess (UDP port {args.port})")
    print("  - Point cloud processor: main thread")
    print("  - Output: print to console (debug mode)")
    print("-" * 50)

    try:
        # 主线程消费循环：从 output_queue 取出处理结果并输出
        # 目前为 debug 模式，仅打印到终端
        # 预留接口：可扩展为保存文件/发送网络/转发到其他系统
        while output_running[0]:
            try:
                result = output_queue.get(timeout=0.5)  # 等待处理结果
                # print(f"[Output] {result}")  # 打印目标航迹信息
            except multiprocessing.queues.Empty:
                continue  # 队列为空，继续等待
            except KeyboardInterrupt:
                break  # 用户中断，退出循环
    finally:
        shutdown()  # 确保资源清理


if __name__ == "__main__":
    main()
