"""
鼠标轨迹收集脚本
实时收集用户的真实鼠标移动轨迹数据，用于训练LSTM模型
"""
import time
import numpy as np
from pynput import mouse
from pynput.mouse import Listener as MouseListener
import os
from datetime import datetime
import json


def get_dpi_scale():
    """
    获取Windows DPI缩放比例
    
    Returns:
        float: DPI缩放比例（例如1.5表示150%）
    """
    try:
        import platform
        if platform.system() == 'Windows':
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            
            # 获取物理分辨率（不受DPI影响）
            hdc = user32.GetDC(0)
            physical_width = gdi32.GetDeviceCaps(hdc, 118)  # HORZRES
            physical_height = gdi32.GetDeviceCaps(hdc, 117)  # VERTRES
            
            # 获取虚拟分辨率（DPI缩放后的）
            virtual_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            virtual_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            user32.ReleaseDC(0, hdc)
            
            # 计算缩放比例
            if physical_width > 0 and virtual_width > 0:
                scale_x = virtual_width / physical_width
                scale_y = virtual_height / physical_height
                # 通常x和y缩放比例相同
                scale = (scale_x + scale_y) / 2.0
                return scale
    except:
        pass
    
    return 1.0  # 默认无缩放


def get_screen_size():
    """
    获取真实的屏幕尺寸（不受DPI缩放影响）
    
    Returns:
        tuple: (width, height) 真实屏幕分辨率
    """
    try:
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
        
        # 备用方法：使用tkinter（可能受DPI影响，需要校正）
        try:
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            
            # 如果检测到DPI缩放，尝试校正
            dpi_scale = get_dpi_scale()
            if dpi_scale > 1.0:
                width = int(width / dpi_scale)
                height = int(height / dpi_scale)
            
            return width, height
        except:
            pass
    except:
        pass
    
    # 如果所有方法都失败，使用默认值
    return 1920, 1080


class MouseTrajectoryCollector:
    """鼠标轨迹收集器"""
    
    def __init__(self, screen_width=1920, screen_height=1080, 
                 min_trajectory_length=10, max_idle_time=2.0,
                 min_distance=10, dpi_scale=None):
        """
        初始化收集器
        
        Args:
            screen_width: 屏幕宽度（像素）- 真实分辨率
            screen_height: 屏幕高度（像素）- 真实分辨率
            min_trajectory_length: 最小轨迹长度（点数）
            max_idle_time: 最大空闲时间（秒），超过此时间认为轨迹结束
            min_distance: 最小移动距离（像素），小于此距离的点会被过滤
            dpi_scale: DPI缩放比例（如果为None则自动检测）
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 检测DPI缩放比例
        if dpi_scale is None:
            self.dpi_scale = get_dpi_scale()
        else:
            self.dpi_scale = dpi_scale
        
        # 计算虚拟分辨率（DPI缩放后的分辨率，这是pynput返回的坐标范围）
        self.virtual_width = int(screen_width * self.dpi_scale)
        self.virtual_height = int(screen_height * self.dpi_scale)
        
        self.min_trajectory_length = min_trajectory_length
        self.max_idle_time = max_idle_time
        self.min_distance = min_distance
        
        # 当前轨迹数据
        self.current_trajectory = []
        self.last_point_time = None
        self.last_point = None
        
        # 所有收集的轨迹
        self.all_trajectories = []
        self.all_targets = []
        
        # 统计信息
        self.total_trajectories = 0
        self.total_points = 0
        self.start_time = None
        
        # 监听器
        self.listener = None
        self.is_collecting = False
        
    def on_move(self, x, y):
        """鼠标移动事件处理"""
        if not self.is_collecting:
            return
        
        current_time = time.time()
        
        current_point = np.array([x, y], dtype=np.float32)
        print(f"\r已收集轨迹: {self.total_trajectories} 条 | "
              f"总点数: {self.total_points} | "
              f"当前轨迹点数: {len(self.current_trajectory)}, 当前位置(真实): {x:.0f},{y:.0f}", end='', flush=True)
        # 检查是否应该开始新轨迹
        if self.last_point_time is None:
            # 第一个点，开始新轨迹
            self.current_trajectory = [current_point]
            self.last_point_time = current_time
            self.last_point = current_point
            return
        
        # 检查是否空闲时间过长（轨迹结束）
        if current_time - self.last_point_time > self.max_idle_time:
            # 保存当前轨迹并开始新轨迹
            self._save_current_trajectory()
            self.current_trajectory = [current_point]
            self.last_point_time = current_time
            self.last_point = current_point
            return
        
        # 计算与上一个点的距离
        distance = np.linalg.norm(current_point - self.last_point)
        
        # 只记录移动距离足够的点（过滤微小抖动）
        if distance >= self.min_distance:
            self.current_trajectory.append(current_point)
            self.last_point_time = current_time
            self.last_point = current_point
            self.total_points += 1
    
    def on_click(self, x, y, button, pressed):
        """鼠标点击事件处理"""
        if not self.is_collecting:
            return
        
        if pressed:
            # 鼠标按下时，如果当前有轨迹，保存它并开始新轨迹
            if len(self.current_trajectory) > 0:
                self._save_current_trajectory()
            
            # 开始新轨迹，以点击位置为起点
            self.current_trajectory = [np.array([x, y], dtype=np.float32)]
            self.last_point_time = time.time()
            self.last_point = np.array([x, y], dtype=np.float32)
        else:
            # 鼠标释放时，如果当前有轨迹，保存它
            if len(self.current_trajectory) > 0:
                self._save_current_trajectory()
    
    def _save_current_trajectory(self):
        """保存当前轨迹"""
        if len(self.current_trajectory) < self.min_trajectory_length:
            # 轨迹太短，丢弃
            self.current_trajectory = []
            return
        
        trajectory = np.array(self.current_trajectory, dtype=np.float32)
        
        # 裁剪坐标到屏幕范围内（防止鼠标移动到屏幕外）
        trajectory[:, 0] = np.clip(trajectory[:, 0], 0, self.screen_width)
        trajectory[:, 1] = np.clip(trajectory[:, 1], 0, self.screen_height)
        
        # 归一化坐标到 [0, 1] 范围
        trajectory_normalized = trajectory.copy()
        trajectory_normalized[:, 0] /= self.screen_width
        trajectory_normalized[:, 1] /= self.screen_height
        
        # 确保归一化坐标在[0, 1]范围内（双重保险）
        trajectory_normalized = np.clip(trajectory_normalized, 0.0, 1.0)
        
        # 起点和目标点
        start_point = trajectory_normalized[0]
        target_point = trajectory_normalized[-1]
        
        # 计算到达目标点后的停留时间
        # 停留时间 = 轨迹总时间（从第一个点到最后一个点的时间）
        stay_time = 0.0
        if len(self.current_timestamps) > 1:
            stay_time = self.current_timestamps[-1] - self.current_timestamps[0]
        
        # 保存轨迹和目标点（目标点包含时间信息：[x, y, time]）
        self.all_trajectories.append(trajectory_normalized)
        # 目标点扩展为3维：[x, y, stay_time]
        target_with_time = np.array([target_point[0], target_point[1], stay_time], dtype=np.float32)
        self.all_targets.append(target_with_time)
        
        self.total_trajectories += 1
        self.current_trajectory = []
        self.current_timestamps = []
        
        # 打印统计信息
        print(f"\r已收集轨迹: {self.total_trajectories} 条 | "
              f"总点数: {self.total_points} | "
              f"当前轨迹点数: {len(trajectory)}", end='', flush=True)
    
    def start_collecting(self):
        """开始收集轨迹"""
        print("=" * 60)
        print("鼠标轨迹收集器")
        print("=" * 60)
        print(f"屏幕尺寸（真实分辨率）: {self.screen_width} x {self.screen_height}")
        print(f"DPI缩放比例: {self.dpi_scale:.2f} ({self.dpi_scale*100:.0f}%)")
        print(f"虚拟分辨率: {self.virtual_width} x {self.virtual_height}")
        print(f"最小轨迹长度: {self.min_trajectory_length} 点")
        print(f"最大空闲时间: {self.max_idle_time} 秒")
        print(f"最小移动距离: {self.min_distance} 像素")
        print("\n开始收集鼠标轨迹...")
        print("提示:")
        print("  - 正常移动鼠标即可收集轨迹")
        print("  - 鼠标静止超过 {} 秒会自动结束当前轨迹".format(self.max_idle_time))
        print("  - 点击鼠标也会结束当前轨迹并开始新轨迹")
        print("  - 按 ESC 键停止收集并保存数据")
        print("-" * 60)
        
        self.is_collecting = True
        self.start_time = time.time()
        
        # 创建鼠标监听器
        self.listener = MouseListener(
            on_move=self.on_move,
            on_click=self.on_click
        )
        self.listener.start()
        
        # 等待用户按ESC键停止
        try:
            with mouse.Listener(on_click=self._on_stop_click) as stop_listener:
                stop_listener.join()
        except KeyboardInterrupt:
            pass
        
        self.stop_collecting()
    
    def _on_stop_click(self, x, y, button, pressed):
        """用于停止收集的点击处理——点击鼠标右键停止收集"""
        from pynput.mouse import Button
        if button == Button.right and pressed:
            print("停止收集")
            # 停止监听用于结束收集
            return False
    
    def stop_collecting(self):
        """停止收集并保存数据"""
        self.is_collecting = False
        
        # 保存最后一个轨迹
        if len(self.current_trajectory) > 0:
            self._save_current_trajectory()
        
        if self.listener:
            self.listener.stop()
        
        print("\n\n" + "=" * 60)
        print("收集完成！")
        print("=" * 60)
        
        if len(self.all_trajectories) == 0:
            print("警告: 未收集到任何轨迹数据")
            return
        
        # 保存数据
        self.save_data()
        
        # 打印统计信息
        elapsed_time = time.time() - self.start_time
        avg_points = np.mean([len(t) for t in self.all_trajectories])
        
        print(f"\n统计信息:")
        print(f"  总轨迹数: {self.total_trajectories}")
        print(f"  总点数: {self.total_points}")
        print(f"  平均轨迹长度: {avg_points:.1f} 点")
        print(f"  收集时间: {elapsed_time:.1f} 秒")
        print(f"  平均速度: {self.total_trajectories / elapsed_time:.2f} 轨迹/秒")
    
    def save_data(self, output_dir="collected_data"):
        """保存收集的数据"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 填充轨迹到相同长度
        max_len = max(len(t) for t in self.all_trajectories)
        padded_trajectories = []
        
        for traj in self.all_trajectories:
            if len(traj) < max_len:
                # 用最后一个点填充
                padding = np.tile(traj[-1:], (max_len - len(traj), 1))
                padded_traj = np.vstack([traj, padding])
            else:
                padded_traj = traj
            padded_trajectories.append(padded_traj)
        
        trajectories_array = np.array(padded_trajectories, dtype=np.float32)
        
        # 处理targets：确保所有targets都是3维（x, y, time）
        processed_targets = []
        for target in self.all_targets:
            if len(target) == 3:
                processed_targets.append(target)
            elif len(target) == 2:
                # 如果只有2维，添加默认时间0
                processed_targets.append(np.array([target[0], target[1], 0.0], dtype=np.float32))
            else:
                # 兼容其他情况
                processed_targets.append(np.array([target[0] if len(target) > 0 else 0.0, 
                                                 target[1] if len(target) > 1 else 0.0, 
                                                 0.0], dtype=np.float32))
        
        targets_array = np.array(processed_targets, dtype=np.float32)
        
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectories_file = os.path.join(output_dir, f"trajectories_{timestamp}.npy")
        targets_file = os.path.join(output_dir, f"targets_{timestamp}.npy")
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        
        # 保存数据
        np.save(trajectories_file, trajectories_array)
        np.save(targets_file, targets_array)
        
        # 保存元数据
        metadata = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "dpi_scale": self.dpi_scale,
            "virtual_width": self.virtual_width,
            "virtual_height": self.virtual_height,
            "num_trajectories": len(self.all_trajectories),
            "max_sequence_length": max_len,
            "min_sequence_length": min(len(t) for t in self.all_trajectories),
            "avg_sequence_length": float(np.mean([len(t) for t in self.all_trajectories])),
            "collection_time": timestamp,
            "total_points": self.total_points
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据已保存:")
        print(f"  轨迹文件: {trajectories_file}")
        print(f"  目标文件: {targets_file}")
        print(f"  元数据文件: {metadata_file}")
        
        return trajectories_file, targets_file, metadata_file


def merge_collected_data(data_dir="collected_data", output_file="trajectories_merged.npy"):
    """合并多个收集的数据文件"""
    import glob
    
    trajectory_files = glob.glob(os.path.join(data_dir, "trajectories_*.npy"))
    target_files = glob.glob(os.path.join(data_dir, "targets_*.npy"))
    
    if not trajectory_files:
        print("未找到收集的数据文件")
        return
    
    all_trajectories = []
    all_targets = []
    
    for traj_file, target_file in zip(sorted(trajectory_files), sorted(target_files)):
        trajectories = np.load(traj_file)
        targets = np.load(target_file)
        all_trajectories.append(trajectories)
        all_targets.append(targets)
        print(f"加载: {os.path.basename(traj_file)} ({len(trajectories)} 条轨迹, 序列长度: {trajectories.shape[1] if len(trajectories.shape) > 1 else 'N/A'})")
    
    # 检查所有轨迹的序列长度
    sequence_lengths = [traj.shape[1] for traj in all_trajectories if len(traj.shape) > 1]
    if not sequence_lengths:
        print("错误: 无法确定轨迹序列长度")
        return
    
    max_len = max(sequence_lengths)
    min_len = min(sequence_lengths)
    print(f"\n序列长度范围: {min_len} - {max_len}")
    
    if max_len != min_len:
        print(f"检测到不同序列长度，将统一填充到最大长度: {max_len}")
    
    # 填充所有轨迹到相同长度
    padded_all_trajectories = []
    for trajectories in all_trajectories:
        if len(trajectories.shape) < 2:
            continue
        
        current_len = trajectories.shape[1]
        if current_len < max_len:
            # 需要填充
            padded_trajectories = []
            for traj in trajectories:
                # 用最后一个点填充
                padding = np.tile(traj[-1:], (max_len - current_len, 1))
                padded_traj = np.vstack([traj, padding])
                padded_trajectories.append(padded_traj)
            padded_all_trajectories.append(np.array(padded_trajectories, dtype=np.float32))
        else:
            padded_all_trajectories.append(trajectories)
    
    # 合并数据
    merged_trajectories = np.vstack(padded_all_trajectories)
    merged_targets = np.vstack(all_targets)

    os.makedirs("merged", exist_ok=True)

    # 保存合并后的数据
    np.save("merged/trajectories.npy", merged_trajectories)
    np.save("merged/targets.npy", merged_targets)
    
    print(f"\n合并完成:")
    print(f"  总轨迹数: {len(merged_trajectories)}")
    print(f"  已保存到: merged/trajectories.npy 和 merged/targets.npy")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="收集鼠标轨迹数据")
    parser.add_argument("--screen-width", type=int, default=None,
                       help="屏幕宽度（像素），默认自动检测")
    parser.add_argument("--screen-height", type=int, default=None,
                       help="屏幕高度（像素），默认自动检测")
    parser.add_argument("--min-length", type=int, default=10,
                       help="最小轨迹长度（点数），默认10")
    parser.add_argument("--max-idle", type=float, default=2.0,
                       help="最大空闲时间（秒），默认2.0")
    parser.add_argument("--min-distance", type=int, default=10,
                       help="最小移动距离（像素），默认10")
    parser.add_argument("--merge", action="store_true",
                       help="合并已收集的数据文件")
    
    args = parser.parse_args()
    
    if args.merge:
        # 合并模式
        merge_collected_data()
    else:
        # 收集模式
        if args.screen_width and args.screen_height:
            screen_width, screen_height = args.screen_width, args.screen_height
        else:
            screen_width, screen_height = get_screen_size()
            print(f"自动检测到屏幕尺寸（真实分辨率）: {screen_width} x {screen_height}")
        
        # 检测DPI缩放
        dpi_scale = get_dpi_scale()
        if dpi_scale != 1.0:
            virtual_width = int(screen_width * dpi_scale)
            virtual_height = int(screen_height * dpi_scale)
            print(f"检测到DPI缩放: {dpi_scale:.2f} ({dpi_scale*100:.0f}%)")
            print(f"虚拟分辨率: {virtual_width} x {virtual_height}")
            print(f"收集的坐标将自动转换为真实分辨率: {screen_width} x {screen_height}")
        
        collector = MouseTrajectoryCollector(
            screen_width=screen_width,
            screen_height=screen_height,
            min_trajectory_length=args.min_length,
            max_idle_time=args.max_idle,
            min_distance=args.min_distance
        )
        
        try:
            collector.start_collecting()
        except KeyboardInterrupt:
            print("\n\n收到中断信号，正在停止收集...")
            collector.stop_collecting()

