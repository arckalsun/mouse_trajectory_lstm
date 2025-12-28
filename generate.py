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

def load_model(model_path='mouse_trajectory_model.pth', device='cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    
    Returns:
        加载的模型
    """
    model = MouseTrajectoryLSTM(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
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
    逐步生成轨迹（更接近真实鼠标移动）
    
    Args:
        model: 训练好的LSTM模型
        target_point: 目标点坐标（归一化）
        screen_width, screen_height: 屏幕尺寸（用于反归一化）
        device: 计算设备
        max_steps: 最大步数
    
    Returns:
        生成的轨迹点数组（屏幕坐标）
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
    current_point = start_point.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
    
    # 维护一个历史窗口（用于LSTM输入）
    history_window = [start_point.cpu().numpy()]
    max_history = 10  # 使用最近10个点作为历史
    
    min_distance = float('inf')
    
    with torch.no_grad():
        for step in range(max_steps):
            # 准备输入序列（使用历史窗口）
            if len(history_window) > 1:
                # 计算相对位移（更有利于模型学习）
                history_array = np.array(history_window[-max_history:])
                if len(history_array) > 1:
                    # 使用相对位移作为输入
                    displacements = np.diff(history_array, axis=0)
                    # 如果只有一个点，添加零位移
                    if len(displacements) == 0:
                        displacements = np.array([[0.0, 0.0]])
                else:
                    displacements = np.array([[0.0, 0.0]])
            else:
                displacements = np.array([[0.0, 0.0]])
            
            # 转换为tensor
            input_seq = torch.FloatTensor(displacements).unsqueeze(0).to(device)
            
            # 预测下一个点的位移和时间间隔
            next_pred, _ = model(input_seq)
            
            # 分离坐标和时间
            if next_pred.shape[1] == 3:  # 包含时间信息
                next_displacement = next_pred[:, :2]
                time_delta = next_pred[:, 2:3].item()
            else:
                next_displacement = next_pred
                time_delta = 0.0
            
            # 计算下一个点
            last_point = torch.FloatTensor(history_window[-1]).to(device)
            next_point = last_point + next_displacement.squeeze()
            
            # 添加朝向目标点的引导
            direction_to_target = target_point - next_point
            distance_to_target = torch.norm(direction_to_target)
            
            # 根据距离调整引导强度
            if distance_to_target > 0.01:
                direction_to_target = direction_to_target / distance_to_target
                # 距离越远，引导越强
                guide_strength = min(0.3, distance_to_target.item() * 2)
                next_point = next_point + guide_strength * direction_to_target * 0.01
            
            # 添加小的随机扰动（模拟人类移动的不规则性）
            noise = torch.randn_like(next_point) * 0.005
            next_point = next_point + noise
            
            # 限制在[0, 1]范围内
            next_point = torch.clamp(next_point, 0.0, 1.0)
            
            next_point_np = next_point.cpu().numpy()
            trajectory.append(next_point_np)
            history_window.append(next_point_np)
            
            # 保持历史窗口大小
            if len(history_window) > max_history:
                history_window.pop(0)
            
            # 检查是否接近目标
            distance = distance_to_target.item()
            min_distance = min(min_distance, distance)
            
            # 停止条件
            if distance < 0.01:  # 归一化坐标下的阈值
                break
            
            # 如果距离不再减小，也停止
            if step > 10 and distance > min_distance * 1.5:
                break
    
    # 反归一化到屏幕坐标
    trajectory_screen = np.array(trajectory) * np.array([screen_width, screen_height])
    
    return trajectory_screen


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
        
        # 生成轨迹
        trajectory = generate_trajectory_step_by_step(
            model=model,
            start_point=np.array(start),
            target_point=np.array(target),
            screen_width=screen_width,
            screen_height=screen_height,
            device=device
        )
        
        print(f"生成了 {len(trajectory)} 个轨迹点")

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

