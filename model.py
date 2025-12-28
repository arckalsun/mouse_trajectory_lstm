"""
LSTM鼠标轨迹生成模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MouseTrajectoryLSTM(nn.Module):
    """
    LSTM模型用于生成鼠标轨迹
    
    输入: 当前轨迹点序列 (batch_size, seq_len, 2)
    输出: 下一个轨迹点和等待时间 (batch_size, 3) - [x, y, time_interval]
    """
    
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, dropout=0.2, 
                 output_time=True):
        """
        Args:
            input_size: 输入特征维度（x, y坐标 = 2）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            output_time: 是否输出时间间隔（等待时间）
        """
        super(MouseTrajectoryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_time = output_time
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 坐标输出层
        self.fc_coord = nn.Linear(hidden_size // 4, 2)  # 输出x, y坐标
        
        # 时间间隔输出层（如果启用）
        if output_time:
            self.fc_time = nn.Sequential(
                nn.Linear(hidden_size // 4, hidden_size // 8),
                nn.ReLU(),
                nn.Linear(hidden_size // 8, 1),  # 输出时间间隔（秒）
                nn.Softplus()  # 确保时间间隔为正数
            )
        else:
            self.fc_time = None
        
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, 2)
            hidden: LSTM的隐藏状态（可选）
        
        Returns:
            output: 预测的下一个点和时间间隔 (batch_size, 3) 或 (batch_size, 2)
            hidden: 新的隐藏状态
        """
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 提取共享特征
        features = self.feature_extractor(last_output)  # (batch_size, hidden_size // 4)
        
        # 预测坐标
        coords = self.fc_coord(features)  # (batch_size, 2)
        
        # 预测时间间隔（如果启用）
        if self.output_time and self.fc_time is not None:
            time_interval = self.fc_time(features)  # (batch_size, 1)
            # 合并坐标和时间间隔
            output = torch.cat([coords, time_interval], dim=1)  # (batch_size, 3)
        else:
            output = coords  # (batch_size, 2)
        
        return output, hidden
    
    def generate_trajectory(self, start_point, target_point, max_steps=200, 
                           device='cpu', temperature=1.0):
        """
        生成从起点到目标点的完整轨迹（包含时间间隔）
        
        Args:
            start_point: 起始点 (2,) numpy array 或 tensor
            target_point: 目标点 (2,) numpy array 或 tensor
            max_steps: 最大步数
            device: 计算设备
            temperature: 控制随机性的温度参数
        
        Returns:
            trajectory: 生成的轨迹点列表 (N, 2) - 仅坐标
            time_intervals: 时间间隔列表 (N-1,) - 每个点之间的等待时间（秒）
        """
        self.eval()
        
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
        hidden = None
        
        # 计算初始距离
        initial_distance = torch.norm(target_point - start_point)
        min_distance = float('inf')
        
        with torch.no_grad():
            for step in range(max_steps):
                # 预测下一个点和时间间隔
                output, hidden = self.forward(current_point, hidden)
                
                # 分离坐标和时间间隔
                if self.output_time and output.shape[1] == 3:
                    next_point_pred = output[:, :2]  # (1, 2)
                    time_interval = output[:, 2].item()  # 标量
                    # 限制时间间隔在合理范围内（0.001秒到0.1秒）
                    time_interval = max(0.001, min(0.1, time_interval))
                    time_intervals.append(time_interval)
                else:
                    next_point_pred = output  # (1, 2)
                    # 如果没有时间输出，使用默认值
                    time_intervals.append(0.01)  # 默认10ms
                
                next_point_pred = next_point_pred.squeeze(0)  # (2,)
                
                # 添加一些朝向目标点的引导（可选）
                # 计算到目标点的方向
                direction_to_target = target_point - current_point.squeeze()
                direction_to_target = direction_to_target / (torch.norm(direction_to_target) + 1e-6)
                
                # 混合预测和方向引导
                blend_factor = 0.1  # 引导强度
                guided_point = next_point_pred + blend_factor * direction_to_target
                
                # 应用温度缩放（增加随机性）
                if temperature > 0:
                    noise = torch.randn_like(guided_point) * 0.01 * temperature
                    guided_point = guided_point + noise
                
                # 更新当前点
                current_point = guided_point.unsqueeze(0).unsqueeze(0)
                trajectory.append(guided_point.cpu().numpy())
                
                # 检查是否接近目标
                distance = torch.norm(target_point - guided_point)
                min_distance = min(min_distance, distance.item())
                
                # 如果足够接近目标，停止生成
                if distance < 0.01:  # 归一化坐标下的阈值
                    break
                
                # 如果距离不再减小，也停止
                if step > 10 and distance > min_distance * 1.5:
                    break
        
        return np.array(trajectory), np.array(time_intervals)

