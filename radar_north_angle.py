import math

def calculate_radar_north_angle(radar_lon, radar_lat, uav_lon, uav_lat, radar_angle):
    """
    计算雷达北向角
    
    参数:
        radar_lon: 雷达经度
        radar_lat: 雷达纬度  
        uav_lon: 无人机经度
        uav_lat: 无人机纬度
        radar_angle: 雷达测量角 (东=90, 北=0, 西=-90)
    
    返回:
        radar_north: 雷达北向角（度）
        true_angle: 真实方位角（度）
    """
    dlon = uav_lon - radar_lon
    dlat = uav_lat - radar_lat
    
    true_angle = math.degrees(math.atan2(dlon, dlat))
    
    radar_north = true_angle - radar_angle
    
    while radar_north > 180:
        radar_north -= 360
    while radar_north < -180:
        radar_north += 360
    
    return radar_north, true_angle


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


if __name__ == "__main__":
    print("=== 雷达北向角计算 ===")
    print("提示: 雷达视角范围 -30° ~ +30°，校准时需让无人机在视野内\n")
    
    radar_lon = float(input("雷达经度: "))
    radar_lat = float(input("雷达纬度: "))
    
    n = int(input("测量点数 (建议3-5点): "))
    
    results = []
    for i in range(n):
        print(f"\n--- 第{i+1}点 ---")
        uav_lon = float(input("无人机经度: "))
        uav_lat = float(input("无人机纬度: "))
        radar_angle = float(input("雷达测量角 (-30 ~ +30): "))
        
        radar_north, true_angle = calculate_radar_north_angle(
            radar_lon, radar_lat, uav_lon, uav_lat, radar_angle
        )
        results.append(radar_north)
        print(f"  真实方位角: {true_angle:.2f}° | 雷达北向角: {radar_north:.2f}°")
    
    avg = sum(results) / len(results)
    avg = normalize_angle(avg)
    if avg < 0:
        avg = 360 + avg
    
    results_360 = [360 + r if r < 0 else r for r in results]
    
    print(f"\n========== 结果 ==========")
    print(f"各次测量北向角: {[f'{r:.1f}°' for r in results_360]}")
    print(f"平均北向角: {avg:.2f}°")
