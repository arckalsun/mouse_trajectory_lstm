"""
使用训练好的LSTM模型生成鼠标轨迹
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MouseTrajectoryLSTM
import os

plt.switch_backend('TkAgg')
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def load_model(model_path='mouse_trajectory_model.pth', device='cpu', output_time=True):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        output_time: 是否启用时间间隔输出
    
    Returns:
        加载的模型
    """
    model = MouseTrajectoryLSTM(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_time=output_time  # 启用时间间隔输出
    )
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型加载成功 (训练轮数: {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"警告: 未找到模型文件 {model_path}，使用未训练的模型")
    
    model.eval()
    return model.to(device)


def generate_trajectory_step_by_step(model, start_point, target_point, 
                                    screen_width=1920, screen_height=1080,
                                    device='cpu', max_steps=200):
    """
    逐步生成轨迹（更接近真实鼠标移动，包含时间间隔）
    
    Args:
        model: 训练好的LSTM模型
        target_point: 目标点坐标（归一化）
        screen_width, screen_height: 屏幕尺寸（用于反归一化）
        device: 计算设备
        max_steps: 最大步数
    
    Returns:
        tuple: (trajectory_screen, time_intervals)
            trajectory_screen: 生成的轨迹点数组（屏幕坐标）(N, 2)
            time_intervals: 时间间隔数组（秒）(N-1,)
    """
    model.eval()
    
    # 转换为tensor
    if isinstance(start_point, np.ndarray):
        start_point = torch.from_numpy(start_point).float()
    if isinstance(target_point, np.ndarray):
        target_point = torch.from_numpy(target_point).float()
    
    start_point = start_point.to(device)
    target_point = target_point.to(device)
    
    trajectory = [start_point.cpu().numpy()]
    time_intervals = []
    current_point = start_point.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
    
    # 维护一个历史窗口（用于LSTM输入）
    history_window = [start_point.cpu().numpy()]
    max_history = 10  # 使用最近10个点作为历史
    
    min_distance = float('inf')
    
    with torch.no_grad():
        for step in range(max_steps):
            # 准备输入序列（使用历史窗口）
            if len(history_window) > 1:
                # 使用历史窗口的坐标作为输入
                history_array = np.array(history_window[-max_history:])
                input_seq = torch.FloatTensor(history_array).unsqueeze(0).to(device)  # (1, seq_len, 2)
            else:
                input_seq = current_point  # (1, 1, 2)
            
            # 预测下一个点和时间间隔
            output, _ = model(input_seq)
            
            # 分离坐标和时间间隔
            if output.shape[1] == 3:
                # 模型输出包含时间间隔
                next_point_pred = output[:, :2].squeeze(0)  # (2,)
                time_interval = output[:, 2].item()  # 标量
                # 限制时间间隔在合理范围内（1ms到100ms）
                time_interval = max(0.001, min(0.1, time_interval))
            else:
                # 旧版本模型，只输出坐标
                next_point_pred = output.squeeze(0)  # (2,)
                time_interval = 0.01  # 默认10ms
            
            time_intervals.append(time_interval)
            
            # 获取当前点（归一化坐标）
            current_point_norm = torch.FloatTensor(history_window[-1]).to(device)
            
            # 计算到目标点的方向和距离
            direction_to_target = target_point - current_point_norm
            distance_to_target = torch.norm(direction_to_target)
            
            # 归一化方向向量
            if distance_to_target > 1e-6:
                direction_to_target = direction_to_target / distance_to_target
            else:
                # 已经到达目标点
                direction_to_target = torch.zeros_like(direction_to_target)
            
            # 混合模型预测和朝向目标的引导
            # 引导强度：距离越远，引导越强；距离越近，引导越弱（更依赖模型预测）
            if distance_to_target > 0.01:
                # 动态调整引导强度：距离越远，引导越强（最大0.5）
                guide_strength = min(0.5, 0.1 + distance_to_target.item() * 0.3)
                # 计算引导步长：根据距离动态调整，但不超过合理范围
                step_size = min(0.05, max(0.01, distance_to_target.item() * 0.1))
                # 混合预测点和引导方向
                guided_offset = guide_strength * direction_to_target * step_size
                next_point = next_point_pred + guided_offset
            else:
                # 接近目标时，直接朝向目标点移动
                step_size = min(0.02, distance_to_target.item())
                next_point = current_point_norm + direction_to_target * step_size
            
            # 添加小的随机扰动（模拟人类移动的不规则性），但扰动要小
            noise_scale = max(0.001, min(0.01, distance_to_target.item() * 0.02))
            noise = torch.randn_like(next_point) * noise_scale
            next_point = next_point + noise
            
            # 限制在[0, 1]范围内
            next_point = torch.clamp(next_point, 0.0, 1.0)
            
            next_point_np = next_point.cpu().numpy()
            trajectory.append(next_point_np)
            history_window.append(next_point_np)
            
            # 保持历史窗口大小
            if len(history_window) > max_history:
                history_window.pop(0)
            
            # 计算当前点到目标点的实际距离（用于停止判断）
            current_to_target = target_point - next_point
            distance = torch.norm(current_to_target).item()
            min_distance = min(min_distance, distance)
            
            # 停止条件：足够接近目标点
            if distance < 0.005:  # 归一化坐标下的阈值（约5像素）
                # 确保最后一个点就是目标点
                trajectory.append(target_point.cpu().numpy())
                break
            
            # 如果距离不再减小且已经移动了足够步数，强制朝向目标点
            if step > 20 and distance > min_distance * 1.2:
                # 如果距离不再减小，直接朝向目标点移动
                remaining_steps = max_steps - step - 1
                if remaining_steps > 0:
                    # 计算剩余步数内需要移动的距离
                    step_to_target = current_to_target / (remaining_steps + 1)
                    next_point = next_point + step_to_target
                    # 继续生成，但强制朝向目标
                else:
                    # 最后一步，直接移动到目标点
                    next_point = target_point
                    trajectory.append(next_point.cpu().numpy())
                    break
    
    # 检查最后一个点是否足够接近目标点，如果不是，添加目标点
    if len(trajectory) > 0:
        last_point = torch.FloatTensor(trajectory[-1]).to(device)
        final_distance = torch.norm(target_point - last_point).item()
        if final_distance > 0.01:  # 如果距离目标点超过1%（归一化坐标）
            # 添加目标点作为最后一个点
            trajectory.append(target_point.cpu().numpy())
            # 如果时间间隔数量不够，添加一个默认值
            if len(time_intervals) < len(trajectory) - 1:
                time_intervals.append(0.01)
    
    # 反归一化到屏幕坐标
    trajectory_screen = np.array(trajectory) * np.array([screen_width, screen_height])
    time_intervals_array = np.array(time_intervals)
    
    return trajectory_screen, time_intervals_array


def visualize_trajectory(trajectory, start_point, target_point, 
                        screen_width=1920, screen_height=1080,
                        save_path=None):
    """
    可视化生成的轨迹
    
    Args:
        trajectory: 轨迹点数组（屏幕坐标）
        start_point: 起始点（归一化坐标）
        target_point: 目标点（归一化坐标）
        screen_width, screen_height: 屏幕尺寸
        save_path: 保存路径（可选）
    """
    # 反归一化起点和终点
    start_screen = np.array(start_point) * np.array([screen_width, screen_height])
    target_screen = np.array(target_point) * np.array([screen_width, screen_height])
    
    plt.figure(figsize=(12, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='生成轨迹')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo', markersize=3, alpha=0.5)
    plt.plot(start_screen[0], start_screen[1], 'go', markersize=15, label='起点')
    plt.plot(target_screen[0], target_screen[1], 'ro', markersize=15, label='目标点')
    
    # 绘制直线路径作为对比
    plt.plot([start_screen[0], target_screen[0]], 
             [start_screen[1], target_screen[1]], 
             'r--', alpha=0.3, label='直线路径')
    
    plt.xlabel('X坐标 (像素)', fontsize=12)
    plt.ylabel('Y坐标 (像素)', fontsize=12)
    plt.title('LSTM生成的鼠标轨迹', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    # 反转Y轴，使其与桌面坐标系一致（Y轴从上到下）
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轨迹图已保存到: {save_path}")
    
    plt.show()


def get_screen_size():
    """
    获取真实的屏幕尺寸（不受DPI缩放影响）

    Returns:
        tuple: (width, height) 真实屏幕分辨率
    """
    import platform
    if platform.system() == 'Windows':
        # Windows系统：使用Windows API获取真实分辨率
        try:
            import ctypes
            from ctypes import wintypes

            # 使用GetDeviceCaps获取真实分辨率
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            hdc = user32.GetDC(0)
            width = gdi32.GetDeviceCaps(hdc, 118)  # HORZRES
            height = gdi32.GetDeviceCaps(hdc, 117)  # VERTRES
            user32.ReleaseDC(0, hdc)

            if width > 0 and height > 0:
                return width, height
        except:
            pass
    print(f"use default screen size")
    # 如果所有方法都失败，使用默认值
    return 1920, 1080


def main():
    """主函数：生成鼠标轨迹"""
    # 检查模型文件
    model_path = 'model/mouse_trajectory_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = load_model(model_path, device)
    
    # 设置屏幕尺寸
    screen_width, screen_height = get_screen_size()
    print(f"屏幕真实分辨率: {screen_width, screen_height }")

    # 示例：生成几个轨迹
    test_cases = [
        # (起点归一化坐标, 终点归一化坐标)
        ([0.1, 0.1], [0.9, 0.9]),
        ([0.5, 0.1], [0.5, 0.9]),
        ([0.1, 0.5], [0.9, 0.5]),
        ([0.2, 0.3], [0.8, 0.7]),
    ]
    
    print("\n开始生成鼠标轨迹...")
    for i, (start, target) in enumerate(test_cases):
        print(f"\n生成轨迹 {i+1}: 起点 {start} -> 终点 {target}")
        
        # 生成轨迹（包含时间间隔）
        trajectory, time_intervals = generate_trajectory_step_by_step(
            model=model,
            start_point=np.array(start),
            target_point=np.array(target),
            screen_width=screen_width,
            screen_height=screen_height,
            device=device
        )
        
        print(f"生成了 {len(trajectory)} 个轨迹点")
        print(f"平均时间间隔: {np.mean(time_intervals)*1000:.2f} ms")
        print(f"总时长: {np.sum(time_intervals)*1000:.2f} ms")

        os.makedirs("generate", exist_ok=True)

        # 可视化
        visualize_trajectory(
            trajectory=trajectory,
            start_point=np.array(start),
            target_point=np.array(target),
            screen_width=screen_width,
            screen_height=screen_height,
            save_path=f'generate/trajectory_{i+1}.png'
        )
    
    print("\n所有轨迹生成完成！")


if __name__ == "__main__":
    main()

