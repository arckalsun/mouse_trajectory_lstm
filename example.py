"""
简单的使用示例
演示如何使用训练好的模型生成鼠标轨迹
"""
import torch
import numpy as np
from generate import load_model, generate_trajectory_step_by_step, visualize_trajectory
import os


def example_single_trajectory():
    """生成单个轨迹的示例"""
    # 检查模型是否存在
    model_path = 'model/mouse_trajectory_model.pth'
    if not os.path.exists(model_path):
        print("错误: 未找到模型文件，请先运行 train.py 训练模型")
        return
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # 定义起点和目标点（归一化坐标，范围[0, 1]）
    start_point = np.array([0.2, 0.3])  # 起点
    target_point = np.array([0.8, 0.7])  # 目标点
    
    print(f"生成轨迹: 起点 {start_point} -> 终点 {target_point}")
    
    # 生成轨迹
    screen_width, screen_height = 2560, 1440
    trajectory, time_intervals_array = generate_trajectory_step_by_step(
        model=model,
        start_point=start_point,
        target_point=target_point,
        screen_width=screen_width,
        screen_height=screen_height,
        device=device,
        max_steps=200
    )
    
    print(f"生成了 {len(trajectory)} 个轨迹点")
    print(f"轨迹点示例（前5个）:")
    for i, point in enumerate(trajectory[:5]):
        print(f"  点 {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    # 可视化
    visualize_trajectory(
        trajectory=trajectory,
        start_point=start_point,
        target_point=target_point,
        screen_width=screen_width,
        screen_height=screen_height,
        save_path='example_trajectory.png'
    )
    
    return trajectory


def example_batch_trajectories():
    """批量生成多个轨迹的示例"""
    model_path = 'mouse_trajectory_model.pth'
    if not os.path.exists(model_path):
        print("错误: 未找到模型文件，请先运行 train.py 训练模型")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # 定义多个起点和目标点对
    test_cases = [
        ([0.1, 0.1], [0.9, 0.9], "左上到右下"),
        ([0.5, 0.1], [0.5, 0.9], "上到下"),
        ([0.1, 0.5], [0.9, 0.5], "左到右"),
        ([0.9, 0.1], [0.1, 0.9], "右上到左下"),
    ]
    
    screen_width, screen_height = 1920, 1080
    trajectories = []
    
    for start, target, desc in test_cases:
        print(f"\n生成轨迹: {desc} ({start} -> {target})")
        trajectory = generate_trajectory_step_by_step(
            model=model,
            start_point=np.array(start),
            target_point=np.array(target),
            screen_width=screen_width,
            screen_height=screen_height,
            device=device
        )
        trajectories.append((trajectory, start, target, desc))
        print(f"  生成了 {len(trajectory)} 个点")
    
    # 可视化所有轨迹
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (trajectory, start, target, desc) in enumerate(trajectories):
        ax = axes[idx]
        start_screen = np.array(start) * np.array([screen_width, screen_height])
        target_screen = np.array(target) * np.array([screen_width, screen_height])
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'bo', markersize=2, alpha=0.5)
        ax.plot(start_screen[0], start_screen[1], 'go', markersize=12, marker='s', label='起点')
        ax.plot(target_screen[0], target_screen[1], 'ro', markersize=12, marker='s', label='目标点')
        ax.plot([start_screen[0], target_screen[0]], 
                [start_screen[1], target_screen[1]], 
                'r--', alpha=0.3, label='直线路径')
        
        ax.set_xlabel('X坐标 (像素)', fontsize=10)
        ax.set_ylabel('Y坐标 (像素)', fontsize=10)
        ax.set_title(f'{desc}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('batch_trajectories.png', dpi=150, bbox_inches='tight')
    print("\n所有轨迹已保存到 batch_trajectories.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM鼠标轨迹生成 - 使用示例")
    print("=" * 60)
    
    # 示例1: 生成单个轨迹
    print("\n【示例1】生成单个轨迹")
    print("-" * 60)
    example_single_trajectory()
    
    # 示例2: 批量生成轨迹
    # print("\n【示例2】批量生成多个轨迹")
    # print("-" * 60)
    # example_batch_trajectories()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

