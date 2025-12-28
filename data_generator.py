"""
鼠标轨迹数据生成器
生成模拟人类鼠标移动轨迹的训练数据
"""
import numpy as np
import math
import random


def generate_human_like_trajectory(start_x, start_y, end_x, end_y, num_points=50):
    """
    生成模拟人类鼠标移动的轨迹
    
    使用贝塞尔曲线和随机扰动来模拟人类鼠标移动的不规则性
    
    Args:
        start_x, start_y: 起始点坐标
        end_x, end_y: 目标点坐标
        num_points: 轨迹点的数量
    
    Returns:
        numpy array: shape (num_points, 2) 的轨迹点数组
    """
    # 计算距离
    distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # 根据距离调整点数和控制点
    if distance < 100:
        num_points = max(20, int(num_points * 0.6))
    elif distance > 1000:
        num_points = int(num_points * 1.5)
    
    # 生成控制点（用于贝塞尔曲线）
    # 控制点会在起点和终点之间随机偏移
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    
    # 添加随机偏移，模拟人类移动的不规则性
    offset_range = distance * 0.3
    ctrl1_x = mid_x + random.uniform(-offset_range, offset_range)
    ctrl1_y = mid_y + random.uniform(-offset_range, offset_range)
    ctrl2_x = mid_x + random.uniform(-offset_range * 0.5, offset_range * 0.5)
    ctrl2_y = mid_y + random.uniform(-offset_range * 0.5, offset_range * 0.5)
    
    # 生成贝塞尔曲线点
    trajectory = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        
        # 三次贝塞尔曲线
        x = (1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * end_x
        y = (1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * end_y
        
        # 添加小幅度随机抖动，模拟手部微颤
        jitter = distance * 0.02
        x += random.gauss(0, jitter)
        y += random.gauss(0, jitter)
        
        trajectory.append([x, y])
    
    # 添加速度变化（人类移动时速度不是恒定的）
    # 开始和结束时较慢，中间较快
    speed_profile = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        # 使用缓入缓出函数
        speed = 3 * t * t - 2 * t * t * t  # smoothstep函数
        speed_profile.append(speed)
    
    # 根据速度曲线重新采样
    if len(trajectory) > 1:
        cumulative_speed = np.cumsum(speed_profile)
        cumulative_speed = cumulative_speed / cumulative_speed[-1]
        
        new_trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # 找到对应的原始轨迹点
            idx = np.searchsorted(cumulative_speed, t)
            idx = min(idx, len(trajectory) - 1)
            new_trajectory.append(trajectory[idx])
        trajectory = new_trajectory
    
    return np.array(trajectory, dtype=np.float32)


def generate_training_data(num_samples=10000, screen_width=1920, screen_height=1080):
    """
    生成训练数据集
    
    Args:
        num_samples: 生成的轨迹样本数量
        screen_width, screen_height: 屏幕尺寸
    
    Returns:
        tuple: (trajectories, targets) 
            trajectories: shape (num_samples, seq_len, 2) 的轨迹序列
            targets: shape (num_samples, 3) 的目标点坐标和时间间隔 [x, y, time]
    """
    trajectories = []
    targets = []
    
    print(f"正在生成 {num_samples} 个训练样本...")
    
    for i in range(num_samples):
        # 随机生成起点和终点
        start_x = random.uniform(0, screen_width)
        start_y = random.uniform(0, screen_height)
        end_x = random.uniform(0, screen_width)
        end_y = random.uniform(0, screen_height)
        
        # 生成轨迹点数量（20-100之间）
        num_points = random.randint(20, 100)
        
        # 生成轨迹
        trajectory = generate_human_like_trajectory(
            start_x, start_y, end_x, end_y, num_points
        )
        
        # 归一化坐标到 [0, 1] 范围
        trajectory[:, 0] /= screen_width
        trajectory[:, 1] /= screen_height
        
        trajectories.append(trajectory)
        
        # 计算轨迹的平均时间间隔（模拟人类移动速度）
        # 假设轨迹总时长在0.1秒到2秒之间，根据距离调整
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        total_time = 0.1 + (distance / screen_width) * 1.5  # 根据距离估算总时间
        avg_time_interval = total_time / num_points
        
        # 目标点包含坐标和时间间隔 [x, y, time]
        targets.append([end_x / screen_width, end_y / screen_height, avg_time_interval])
        
        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")
    
    # 填充到相同长度（使用最后一个点填充）
    max_len = max(len(t) for t in trajectories)
    padded_trajectories = []
    
    for traj in trajectories:
        if len(traj) < max_len:
            # 用最后一个点填充
            padding = np.tile(traj[-1:], (max_len - len(traj), 1))
            padded_traj = np.vstack([traj, padding])
        else:
            padded_traj = traj
        padded_trajectories.append(padded_traj)
    
    return np.array(padded_trajectories, dtype=np.float32), np.array(targets, dtype=np.float32)


if __name__ == "__main__":
    # 生成测试数据
    trajectories, targets = generate_training_data(num_samples=1000)
    print(f"轨迹形状: {trajectories.shape}")
    print(f"目标形状: {targets.shape}")
    
    # 保存数据
    np.save("trained_data/trajectories.npy", trajectories)
    np.save("trained_data/targets.npy", targets)
    print("数据已保存到 trajectories.npy 和 targets.npy")

