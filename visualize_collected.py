"""
可视化收集到的鼠标轨迹数据
展示收集到的坐标点和鼠标移动轨迹
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())

# 尝试导入 mplcursors 用于交互式显示
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False
    print("提示: 安装 mplcursors 可以获得更好的交互体验: pip install mplcursors")

# 配置中文字体
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


def load_collected_data(data_dir="collected_data", timestamp=None):
    """
    加载收集的轨迹数据
    
    Args:
        data_dir: 数据目录
        timestamp: 时间戳（可选），如果为None则加载最新的数据
    
    Returns:
        tuple: (trajectories, targets, metadata)
            trajectories: 轨迹数组 (num_trajectories, seq_len, 2)，归一化坐标
            targets: 目标点数组 (num_trajectories, 2) 或 (num_trajectories, 3)
                     2维格式: [x, y]，3维格式: [x, y, time_interval]
            metadata: 元数据字典
    """
    if timestamp is None:
        # 查找最新的数据文件
        trajectory_files = glob.glob(os.path.join(data_dir, "trajectories_*.npy"))
        if not trajectory_files:
            raise FileNotFoundError(f"未找到数据文件在目录: {data_dir}")
        
        # 获取最新的文件
        latest_file = max(trajectory_files, key=os.path.getctime)
        timestamp = Path(latest_file).stem.split("_", 1)[1]
        print(f"加载最新数据: {timestamp}")
    else:
        latest_file = os.path.join(data_dir, f"trajectories_{timestamp}.npy")
        if not os.path.exists(latest_file):
            raise FileNotFoundError(f"未找到数据文件: {latest_file}")
    
    # 加载数据
    trajectories_file = os.path.join(data_dir, f"trajectories_{timestamp}.npy")
    targets_file = os.path.join(data_dir, f"targets_{timestamp}.npy")
    metadata_file = os.path.join(data_dir, f"metadata_{timestamp}.json")
    
    trajectories = np.load(trajectories_file)
    targets = np.load(targets_file)
    
    # 检查targets格式并打印信息
    if len(targets.shape) == 2:
        if targets.shape[1] == 3:
            print(f"检测到3维targets格式 (包含时间间隔): shape {targets.shape}")
        elif targets.shape[1] == 2:
            print(f"检测到2维targets格式: shape {targets.shape}")
        else:
            print(f"警告: targets格式异常: shape {targets.shape}")
    
    # 加载元数据
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return trajectories, targets, metadata


def remove_padding(trajectory, tolerance=1e-6):
    """
    移除轨迹中的填充点（重复的最后一个点）
    
    Args:
        trajectory: 轨迹数组 (seq_len, 2)，归一化坐标
        tolerance: 判断点是否重复的容差
    
    Returns:
        去除填充后的轨迹（归一化坐标）
    """
    if len(trajectory) <= 1:
        return trajectory
    
    # 从后往前找到第一个与最后一个点不同的点
    last_point = trajectory[-1]
    valid_end_idx = len(trajectory) - 1
    
    for i in range(len(trajectory) - 2, -1, -1):
        if not np.allclose(trajectory[i], last_point, atol=tolerance):
            valid_end_idx = i + 1
            break
    else:
        # 如果所有点都相同，只保留第一个点
        return trajectory[:1]
    
    return trajectory[:valid_end_idx + 1]


def clip_coordinates(coords, screen_width, screen_height):
    """
    裁剪坐标到屏幕范围内
    
    Args:
        coords: 坐标数组 (N, 2)，屏幕坐标
        screen_width: 屏幕宽度
        screen_height: 屏幕高度
    
    Returns:
        裁剪后的坐标数组
    """
    coords = coords.copy()
    coords[:, 0] = np.clip(coords[:, 0], 0, screen_width)
    coords[:, 1] = np.clip(coords[:, 1], 0, screen_height)
    return coords


def calculate_velocity_profile(trajectory_screen):
    """
    计算轨迹的速度变化
    
    Args:
        trajectory_screen: 轨迹点数组 (N, 2)，屏幕坐标
    
    Returns:
        tuple: (distances, velocities, cumulative_distance)
            distances: 相邻点之间的距离数组 (N-1,)
            velocities: 速度数组（像素/点），假设每点时间间隔相同 (N-1,)
            cumulative_distance: 累积距离数组 (N,)
    """
    if len(trajectory_screen) < 2:
        return np.array([]), np.array([]), np.array([0.0])
    
    # 计算相邻点之间的距离
    diffs = np.diff(trajectory_screen, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    # 假设每个点的时间间隔相同，速度 = 距离 / 时间间隔
    # 由于时间间隔相同，我们可以直接用距离作为速度的度量
    # 或者可以计算瞬时速度（距离的变化率）
    velocities = distances.copy()
    
    # 计算累积距离
    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
    
    return distances, velocities, cumulative_distance


def visualize_all_trajectories(trajectories, targets, metadata, 
                               max_trajectories=None, save_path=None):
    """
    可视化所有收集的轨迹
    
    Args:
        trajectories: 轨迹数组 (num_trajectories, seq_len, 2)，归一化坐标
        targets: 目标点数组 (num_trajectories, 2) 或 (num_trajectories, 3)
                 2维格式: [x, y]，3维格式: [x, y, time_interval]，归一化坐标
        metadata: 元数据字典
        max_trajectories: 最多显示的轨迹数（None表示显示全部）
        save_path: 保存路径（可选）
    """
    screen_width = metadata.get('screen_width', 1920)
    screen_height = metadata.get('screen_height', 1080)
    
    num_trajectories = len(trajectories)
    if max_trajectories is not None:
        num_trajectories = min(num_trajectories, max_trajectories)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 为每条轨迹分配不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, num_trajectories))
    
    # 存储所有轨迹数据用于交互显示
    all_trajectories_data = []
    
    print(f"正在可视化 {num_trajectories} 条轨迹...")
    
    for i in range(num_trajectories):
        # 移除填充点
        traj = remove_padding(trajectories[i])
        
        # 检查归一化坐标是否在有效范围内，如果超出则裁剪
        traj = np.clip(traj, 0.0, 1.0)
        
        # 转换为屏幕坐标
        traj_screen = traj * np.array([screen_width, screen_height])
        
        # 裁剪到屏幕范围内（防止越界）
        traj_screen = clip_coordinates(traj_screen, screen_width, screen_height)
        
        start_screen = traj_screen[0]
        # 处理targets可能是2维或3维的情况（3维包含时间间隔）
        target_coord = targets[i][:2] if len(targets[i]) >= 2 else targets[i]
        target_screen = np.clip(target_coord * np.array([screen_width, screen_height]), 
                               [0, 0], [screen_width, screen_height])
        
        # 绘制轨迹
        color = colors[i]
        line = ax.plot(traj_screen[:, 0], traj_screen[:, 1], 
                      color=color, linewidth=1.5, alpha=0.6, 
                      label=f'轨迹 {i+1}' if i < 10 else None,
                      picker=True, pickradius=5)[0]
        
        # 存储轨迹数据用于交互
        all_trajectories_data.append({
            'line': line,
            'points': traj_screen,
            'traj_idx': i
        })
        
        # 绘制轨迹点
        scatter = ax.scatter(traj_screen[:, 0], traj_screen[:, 1], 
                            c=[color], s=10, alpha=0.4, edgecolors='none',
                            picker=True, pickradius=5)
        
        # 标记起点
        ax.scatter(start_screen[0], start_screen[1], 
                  c='green', s=100, marker='s', 
                  edgecolors='black', linewidths=1, 
                  zorder=5, label='起点' if i == 0 else None)
        
        # 标记终点
        ax.scatter(target_screen[0], target_screen[1], 
                  c='red', s=100, marker='s', 
                  edgecolors='black', linewidths=1, 
                  zorder=5, label='终点' if i == 0 else None)
    
    ax.set_xlabel('X坐标 (像素)', fontsize=12)
    ax.set_ylabel('Y坐标 (像素)', fontsize=12)
    ax.set_title(f'收集到的鼠标轨迹 (共 {len(trajectories)} 条)', 
                fontsize=14, fontweight='bold')
    # 将图例放在坐标系下方，不遮挡轨迹
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
             ncol=min(5, num_trajectories), fontsize=8, 
             framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    # 设置坐标轴范围，与屏幕坐标系一致（原点在左上角，Y轴向下）
    # 添加边距以便查看边界情况（屏幕尺寸的5%作为边距）
    margin_x = screen_width * 0.05
    margin_y = screen_height * 0.05
    ax.set_xlim(-margin_x, screen_width + margin_x)
    ax.set_ylim(screen_height + margin_y, -margin_y)  # 翻转Y轴，使原点在左上角
    ax.set_aspect('equal', adjustable='box')
    # 将X轴移到上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # 添加鼠标悬停交互功能
    if HAS_MPLCURSORS:
        # 使用 mplcursors 实现交互式显示
        cursor = mplcursors.cursor(ax, hover=True)
        
        @cursor.connect("add")
        def on_add(sel):
            # 找到最近的轨迹点
            x, y = sel.target
            min_dist = float('inf')
            nearest_point = None
            nearest_traj_idx = None
            
            for traj_data in all_trajectories_data:
                points = traj_data['points']
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
                min_idx = np.argmin(distances)
                dist = distances[min_idx]
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = points[min_idx]
                    nearest_traj_idx = traj_data['traj_idx']
            
            if nearest_point is not None and min_dist < 50:  # 50像素范围内
                sel.annotation.set_text(
                    f'轨迹 {nearest_traj_idx + 1}\n'
                    f'坐标: ({nearest_point[0]:.0f}, {nearest_point[1]:.0f})'
                )
            else:
                sel.annotation.set_text(f'坐标: ({x:.0f}, {y:.0f})')
    else:
        # 使用事件处理实现交互
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), 
                           textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        
        def update_annot(event):
            if event.inaxes != ax:
                return
            
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            
            # 找到最近的轨迹点
            min_dist = float('inf')
            nearest_point = None
            nearest_traj_idx = None
            
            for traj_data in all_trajectories_data:
                points = traj_data['points']
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
                min_idx = np.argmin(distances)
                dist = distances[min_idx]
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = points[min_idx]
                    nearest_traj_idx = traj_data['traj_idx']
            
            if nearest_point is not None and min_dist < 50:  # 50像素范围内
                annot.xy = (nearest_point[0], nearest_point[1])
                text = (f'轨迹 {nearest_traj_idx + 1}\n'
                       f'坐标: ({nearest_point[0]:.0f}, {nearest_point[1]:.0f})')
            else:
                annot.xy = (x, y)
                text = f'坐标: ({x:.0f}, {y:.0f})'
            
            annot.set_text(text)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        
        def on_mouse_move(event):
            update_annot(event)
        
        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def visualize_trajectory_detail(trajectories, targets, metadata, 
                                trajectory_indices=None, save_path=None):
    """
    详细可视化指定的轨迹
    
    Args:
        trajectories: 轨迹数组 (num_trajectories, seq_len, 2)，归一化坐标
        targets: 目标点数组 (num_trajectories, 2) 或 (num_trajectories, 3)
                 2维格式: [x, y]，3维格式: [x, y, time_interval]，归一化坐标
        metadata: 元数据字典
        trajectory_indices: 要显示的轨迹索引列表（None表示显示前几条）
        save_path: 保存路径（可选）
    """
    screen_width = metadata.get('screen_width', 1920)
    screen_height = metadata.get('screen_height', 1080)
    
    if trajectory_indices is None:
        trajectory_indices = list(range(min(4, len(trajectories))))
    
    num_trajectories = len(trajectory_indices)
    cols = 2
    rows = (num_trajectories + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    if num_trajectories == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, traj_idx in enumerate(trajectory_indices):
        ax = axes[idx]
        
        # 移除填充点
        traj = remove_padding(trajectories[traj_idx])
        
        # 检查归一化坐标是否在有效范围内，如果超出则裁剪
        traj = np.clip(traj, 0.0, 1.0)
        
        # 转换为屏幕坐标
        traj_screen = traj * np.array([screen_width, screen_height])
        
        # 裁剪到屏幕范围内（防止越界）
        traj_screen = clip_coordinates(traj_screen, screen_width, screen_height)
        
        start_screen = traj_screen[0]
        # 处理targets可能是2维或3维的情况（3维包含时间间隔）
        target_coord = targets[traj_idx][:2] if len(targets[traj_idx]) >= 2 else targets[traj_idx]
        target_screen = np.clip(target_coord * np.array([screen_width, screen_height]),
                               [0, 0], [screen_width, screen_height])
        
        # 绘制轨迹线
        line = ax.plot(traj_screen[:, 0], traj_screen[:, 1], 
                      'b-', linewidth=2, alpha=0.7, label='轨迹路径',
                      picker=True, pickradius=5)[0]
        
        # 绘制所有轨迹点
        scatter = ax.scatter(traj_screen[:, 0], traj_screen[:, 1], 
                            c='blue', s=20, alpha=0.5, edgecolors='none', 
                            label='轨迹点', picker=True, pickradius=5)
        
        # 添加交互功能（使用闭包确保每个子图有独立的变量）
        def make_interactive(ax, traj_screen, traj_idx):
            if HAS_MPLCURSORS:
                cursor = mplcursors.cursor(ax, hover=True)
                
                @cursor.connect("add")
                def on_add(sel):
                    x, y = sel.target
                    # 找到最近的轨迹点
                    distances = np.sqrt((traj_screen[:, 0] - x)**2 + (traj_screen[:, 1] - y)**2)
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    
                    if min_dist < 50:  # 50像素范围内
                        point = traj_screen[min_idx]
                        sel.annotation.set_text(
                            f'轨迹 {traj_idx + 1}\n'
                            f'点 {min_idx + 1}/{len(traj_screen)}\n'
                            f'坐标: ({point[0]:.0f}, {point[1]:.0f})'
                        )
                    else:
                        sel.annotation.set_text(f'坐标: ({x:.0f}, {y:.0f})')
            else:
                # 使用事件处理
                annot = ax.annotate("", xy=(0,0), xytext=(20,20), 
                                   textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                   arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)
                
                def update_annot(event):
                    if event.inaxes != ax:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                        return
                    
                    x, y = event.xdata, event.ydata
                    if x is None or y is None:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                        return
                    
                    distances = np.sqrt((traj_screen[:, 0] - x)**2 + (traj_screen[:, 1] - y)**2)
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    
                    if min_dist < 50:
                        point = traj_screen[min_idx]
                        annot.xy = (point[0], point[1])
                        text = (f'轨迹 {traj_idx + 1}\n'
                               f'点 {min_idx + 1}/{len(traj_screen)}\n'
                               f'坐标: ({point[0]:.0f}, {point[1]:.0f})')
                    else:
                        annot.xy = (x, y)
                        text = f'坐标: ({x:.0f}, {y:.0f})'
                    
                    annot.set_text(text)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                
                fig.canvas.mpl_connect("motion_notify_event", update_annot)
        
        make_interactive(ax, traj_screen, traj_idx)
        
        # 标记起点
        ax.scatter(start_screen[0], start_screen[1], 
                  c='green', s=200, marker='s', 
                  edgecolors='black', linewidths=2, 
                  zorder=5, label='起点')
        
        # 标记终点
        ax.scatter(target_screen[0], target_screen[1], 
                  c='red', s=200, marker='s', 
                  edgecolors='black', linewidths=2, 
                  zorder=5, label='终点')
        
        # 绘制直线路径作为对比
        ax.plot([start_screen[0], target_screen[0]], 
               [start_screen[1], target_screen[1]], 
               'r--', alpha=0.3, linewidth=1, label='直线路径')
        
        # 计算轨迹统计信息
        distances = np.linalg.norm(np.diff(traj_screen, axis=0), axis=1)
        total_distance = np.sum(distances)
        straight_distance = np.linalg.norm(target_screen - start_screen)
        efficiency = straight_distance / total_distance if total_distance > 0 else 0
        
        ax.set_xlabel('X坐标 (像素)', fontsize=10)
        ax.set_ylabel('Y坐标 (像素)', fontsize=10)
        ax.set_title(f'轨迹 {traj_idx+1} (点数: {len(traj)}, '
                    f'效率: {efficiency:.2%})', 
                    fontsize=12, fontweight='bold')
        # 将图例放在坐标系下方
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 ncol=4, fontsize=8, framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        # 设置坐标轴范围，与屏幕坐标系一致（原点在左上角，Y轴向下）
        # 添加边距以便查看边界情况（屏幕尺寸的5%作为边距）
        margin_x = screen_width * 0.05
        margin_y = screen_height * 0.05
        ax.set_xlim(-margin_x, screen_width + margin_x)
        ax.set_ylim(screen_height + margin_y, -margin_y)  # 翻转Y轴，使原点在左上角
        ax.set_aspect('equal', adjustable='box')
        # 将X轴移到上方
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    
    # 隐藏多余的子图
    for idx in range(num_trajectories, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"详细图表已保存到: {save_path}")
    
    plt.show()


def print_statistics(trajectories, targets, metadata):
    """打印收集数据的统计信息"""
    print("\n" + "=" * 60)
    print("收集数据统计信息")
    print("=" * 60)
    
    screen_width = metadata.get('screen_width', 1920)
    screen_height = metadata.get('screen_height', 1080)
    
    print(f"屏幕尺寸: {screen_width} x {screen_height}")
    print(f"轨迹数量: {len(trajectories)}")
    print(f"最大序列长度: {metadata.get('max_sequence_length', 'N/A')}")
    print(f"最小序列长度: {metadata.get('min_sequence_length', 'N/A')}")
    print(f"平均序列长度: {metadata.get('avg_sequence_length', 0):.1f}")
    print(f"总点数: {metadata.get('total_points', 'N/A')}")
    print(f"收集时间: {metadata.get('collection_time', 'N/A')}")
    
    # 计算轨迹效率统计
    efficiencies = []
    distances = []
    out_of_bounds_count = 0
    
    for i in range(len(trajectories)):
        traj = remove_padding(trajectories[i])
        
        # 检查并裁剪归一化坐标
        traj_original = traj.copy()
        traj = np.clip(traj, 0.0, 1.0)
        
        # 检查是否有越界点
        if not np.allclose(traj, traj_original):
            out_of_bounds_count += 1
        
        traj_screen = traj * np.array([screen_width, screen_height])
        traj_screen = clip_coordinates(traj_screen, screen_width, screen_height)
        
        start_screen = traj_screen[0]
        # 处理targets可能是2维或3维的情况（3维包含时间间隔）
        target_coord = targets[i][:2] if len(targets[i]) >= 2 else targets[i]
        target_screen = np.clip(target_coord * np.array([screen_width, screen_height]),
                               [0, 0], [screen_width, screen_height])
        
        # 计算实际移动距离
        traj_distances = np.linalg.norm(np.diff(traj_screen, axis=0), axis=1)
        total_distance = np.sum(traj_distances)
        straight_distance = np.linalg.norm(target_screen - start_screen)
        
        if total_distance > 0:
            efficiency = straight_distance / total_distance
            efficiencies.append(efficiency)
            distances.append(total_distance)
    
    if efficiencies:
        print(f"\n轨迹效率统计:")
        print(f"  平均效率: {np.mean(efficiencies):.2%}")
        print(f"  最高效率: {np.max(efficiencies):.2%}")
        print(f"  最低效率: {np.min(efficiencies):.2%}")
        print(f"  平均移动距离: {np.mean(distances):.1f} 像素")
        print(f"  最大移动距离: {np.max(distances):.1f} 像素")
        print(f"  最小移动距离: {np.min(distances):.1f} 像素")
    
    if out_of_bounds_count > 0:
        print(f"\n警告: 发现 {out_of_bounds_count} 条轨迹包含越界点，已自动裁剪")
    
    print("=" * 60 + "\n")


def visualize_velocity_profile(trajectories, targets, metadata,
                               trajectory_indices=None, save_path=None):
    """
    可视化轨迹的速度变化
    
    Args:
        trajectories: 轨迹数组
        targets: 目标点数组
        metadata: 元数据字典
        trajectory_indices: 要显示的轨迹索引列表（None表示显示前几条）
        save_path: 保存路径（可选）
    """
    screen_width = metadata.get('screen_width', 1920)
    screen_height = metadata.get('screen_height', 1080)
    
    if trajectory_indices is None:
        trajectory_indices = list(range(min(4, len(trajectories))))
    
    num_trajectories = len(trajectory_indices)
    cols = 2
    rows = (num_trajectories + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if num_trajectories == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, traj_idx in enumerate(trajectory_indices):
        ax = axes[idx]
        
        # 移除填充点
        traj = remove_padding(trajectories[traj_idx])
        
        # 检查归一化坐标是否在有效范围内，如果超出则裁剪
        traj = np.clip(traj, 0.0, 1.0)
        
        # 转换为屏幕坐标
        traj_screen = traj * np.array([screen_width, screen_height])
        
        # 裁剪到屏幕范围内（防止越界）
        traj_screen = clip_coordinates(traj_screen, screen_width, screen_height)
        
        # 计算速度变化
        distances, velocities, cumulative_distance = calculate_velocity_profile(traj_screen)
        
        if len(velocities) == 0:
            ax.text(0.5, 0.5, '轨迹点数不足', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'轨迹 {traj_idx+1} - 速度变化', fontsize=12, fontweight='bold')
            continue
        
        # 绘制速度曲线（相对于累积距离）
        # 使用累积距离的中点作为x轴位置
        if len(cumulative_distance) > 1:
            mid_points = (cumulative_distance[:-1] + cumulative_distance[1:]) / 2
        else:
            mid_points = cumulative_distance
        
        ax.plot(mid_points, velocities, 'b-', linewidth=2, alpha=0.7, label='速度')
        ax.fill_between(mid_points, 0, velocities, alpha=0.3, color='blue')
        
        # 添加移动平均线以显示趋势
        if len(velocities) > 5:
            window_size = min(5, len(velocities) // 4)
            if window_size > 1:
                # 计算移动平均
                kernel = np.ones(window_size) / window_size
                smoothed = np.convolve(velocities, kernel, mode='same')
                ax.plot(mid_points, smoothed, 'r--', linewidth=2, alpha=0.8, 
                       label=f'移动平均 (窗口={window_size})')
        
        # 标记起点和终点
        if len(cumulative_distance) > 0:
            ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1, label='起点')
            ax.axvline(x=cumulative_distance[-1], color='red', linestyle='--', 
                      alpha=0.5, linewidth=1, label='终点')
        
        ax.set_xlabel('累积距离 (像素)', fontsize=10)
        ax.set_ylabel('速度 (像素/点)', fontsize=10)
        ax.set_title(f'轨迹 {traj_idx+1} - 速度变化 (总距离: {cumulative_distance[-1]:.0f} 像素, '
                    f'平均速度: {np.mean(velocities):.1f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = (f'最大速度: {np.max(velocities):.1f}\n'
                     f'最小速度: {np.min(velocities):.1f}\n'
                     f'速度标准差: {np.std(velocities):.1f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(num_trajectories, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"速度变化图表已保存到: {save_path}")
    
    plt.show()


def visualize_velocity_comparison(trajectories, targets, metadata,
                                  trajectory_indices=None, save_path=None):
    """
    对比多条轨迹的速度变化
    
    Args:
        trajectories: 轨迹数组
        targets: 目标点数组
        metadata: 元数据字典
        trajectory_indices: 要显示的轨迹索引列表（None表示显示前几条）
        save_path: 保存路径（可选）
    """
    screen_width = metadata.get('screen_width', 1920)
    screen_height = metadata.get('screen_height', 1080)
    
    if trajectory_indices is None:
        trajectory_indices = list(range(min(6, len(trajectories))))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_indices)))
    
    for idx, traj_idx in enumerate(trajectory_indices):
        # 移除填充点
        traj = remove_padding(trajectories[traj_idx])
        traj = np.clip(traj, 0.0, 1.0)
        
        # 转换为屏幕坐标
        traj_screen = traj * np.array([screen_width, screen_height])
        traj_screen = clip_coordinates(traj_screen, screen_width, screen_height)
        
        # 计算速度变化
        distances, velocities, cumulative_distance = calculate_velocity_profile(traj_screen)
        
        if len(velocities) == 0:
            continue
        
        # 归一化累积距离到[0, 1]以便对比
        if cumulative_distance[-1] > 0:
            normalized_distance = cumulative_distance / cumulative_distance[-1]
        else:
            normalized_distance = cumulative_distance
        
        # 使用归一化距离的中点
        if len(normalized_distance) > 1:
            mid_points = (normalized_distance[:-1] + normalized_distance[1:]) / 2
        else:
            mid_points = normalized_distance
        
        color = colors[idx]
        ax.plot(mid_points, velocities, color=color, linewidth=2, alpha=0.7,
               label=f'轨迹 {traj_idx+1}')
    
    ax.set_xlabel('归一化距离 (0=起点, 1=终点)', fontsize=12)
    ax.set_ylabel('速度 (像素/点)', fontsize=12)
    ax.set_title('轨迹速度对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"速度对比图表已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化收集的鼠标轨迹数据")
    parser.add_argument("--data-dir", type=str, default="collected_data",
                       help="数据目录，默认 collected_data")
    parser.add_argument("--timestamp", type=str, default=None,
                       help="数据时间戳（可选），默认使用最新的数据")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "detail", "both", "velocity", "velocity-compare"],
                       help="可视化模式: all(所有轨迹), detail(详细), both(两者), "
                            "velocity(速度变化), velocity-compare(速度对比)")
    parser.add_argument("--max-trajectories", type=int, default=None,
                       help="最多显示的轨迹数（仅用于all模式）")
    parser.add_argument("--trajectory-indices", type=int, nargs="+", default=None,
                       help="要详细显示的轨迹索引（用于detail模式）")
    parser.add_argument("--save-all", type=str, default=None,
                       help="保存所有轨迹图表的路径")
    parser.add_argument("--save-detail", type=str, default=None,
                       help="保存详细轨迹图表的路径")
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print("正在加载数据...")
        trajectories, targets, metadata = load_collected_data(
            data_dir=args.data_dir,
            timestamp=args.timestamp
        )
        
        print(f"成功加载 {len(trajectories)} 条轨迹")
        
        # 打印统计信息
        print_statistics(trajectories, targets, metadata)

        os.makedirs("collected_trajectories", exist_ok=True)
        # 可视化
        if args.mode in ["all", "both"]:
            print("\n可视化所有轨迹...")
            visualize_all_trajectories(
                trajectories, targets, metadata,
                max_trajectories=args.max_trajectories,
                save_path=args.save_all or "collected_trajectories/collected_trajectories_all.png"
            )
        
        if args.mode in ["detail", "both"]:
            print("\n详细可视化轨迹...")
            visualize_trajectory_detail(
                trajectories, targets, metadata,
                trajectory_indices=args.trajectory_indices,
                save_path=args.save_detail or "collected_trajectories/collected_trajectories_detail.png"
            )
        
        if args.mode == "velocity":
            print("\n可视化速度变化...")
            visualize_velocity_profile(
                trajectories, targets, metadata,
                trajectory_indices=args.trajectory_indices,
                save_path="collected_trajectories/velocity_profile.png"
            )
        
        if args.mode == "velocity-compare":
            print("\n对比轨迹速度...")
            visualize_velocity_comparison(
                trajectories, targets, metadata,
                trajectory_indices=args.trajectory_indices,
                save_path="collected_trajectories/velocity_comparison.png"
            )
        
        print("\n可视化完成！")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 collect_trajectory.py 收集数据")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

