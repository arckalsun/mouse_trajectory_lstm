"""
LSTM鼠标轨迹生成模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MouseTrajectoryLSTM(nn.Module):
    """
    LSTM模型用于生成鼠标轨迹（包含时间信息）
    
    输入: 当前轨迹点序列 (batch_size, seq_len, 2)
    输出: 下一个轨迹点和时间间隔 (batch_size, 3) - [x, y, time_delta]
    """
    
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, dropout=0.2, output_time=True):
        """
        Args:
            input_size: 输入特征维度（x, y坐标 = 2）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            output_time: 是否输出时间信息（默认True）
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
        
        # 输出维度：如果包含时间则为3（x, y, time_delta），否则为2（x, y）
        output_dim = 3 if output_time else 2
        
        # 全连接层：将LSTM输出映射到坐标和时间
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_dim)  # 输出x, y坐标和时间间隔
        )
        
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, 2)
            hidden: LSTM的隐藏状态（可选）
        
        Returns:
            output: 预测的下一个点和时间间隔 (batch_size, 3) - [x, y, time_delta]
                    如果output_time=False，则为 (batch_size, 2) - [x, y]
            hidden: 新的隐藏状态
        """
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 通过全连接层得到坐标和时间
        output = self.fc(last_output)  # (batch_size, 3) 或 (batch_size, 2)
        
        # 确保时间间隔为非负值（使用ReLU或softplus）
        if self.output_time and output.shape[1] == 3:
            # 对时间间隔应用softplus确保为正数
            output_time = torch.nn.functional.softplus(output[:, 2:3])
            output = torch.cat([output[:, :2], output_time], dim=1)
        
        return output, hidden
    
    def generate_trajectory(self, start_point, target_point, max_steps=200, 
                           device='cpu', temperature=1.0):
        """
        生成从起点到目标点的完整轨迹（包含时间信息）
        
        Args:
            start_point: 起始点 (2,) numpy array 或 tensor
            target_point: 目标点 (2,) numpy array 或 tensor
            max_steps: 最大步数
            device: 计算设备
            temperature: 控制随机性的温度参数
        
        Returns:
            tuple: (trajectory, time_deltas, total_time)
                trajectory: 生成的轨迹点列表 (N, 2)
                time_deltas: 每个点的时间间隔列表 (N-1,)
                total_time: 总时间（秒）
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
        time_deltas = []
        current_point = start_point.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        hidden = None
        
        # 计算初始距离
        initial_distance = torch.norm(target_point - start_point)
        min_distance = float('inf')
        
        # 时间归一化参数（从训练数据中学习，这里使用默认值）
        max_time = 5.0  # 假设最大停留时间为5秒
        
        with torch.no_grad():
            for step in range(max_steps):
                # 预测下一个点和时间间隔
                next_pred, hidden = self.forward(current_point, hidden)
                
                # 分离坐标和时间
                if next_pred.shape[1] == 3:  # 包含时间信息
                    next_point_pred = next_pred[:, :2]
                    time_delta_normalized = next_pred[:, 2:3].item()
                    # 反归一化时间（从[0,1]恢复到实际秒数）
                    time_delta = time_delta_normalized * max_time
                else:
                    next_point_pred = next_pred
                    time_delta = 0.0
                
                # 添加一些朝向目标点的引导（可选）
                # 计算到目标点的方向
                direction_to_target = target_point - current_point.squeeze()
                direction_to_target = direction_to_target / (torch.norm(direction_to_target) + 1e-6)
                
                # 混合预测和方向引导
                blend_factor = 0.1  # 引导强度
                guided_point = next_point_pred.squeeze() + blend_factor * direction_to_target
                
                # 应用温度缩放（增加随机性）
                if temperature > 0:
                    noise = torch.randn_like(guided_point) * 0.01 * temperature
                    guided_point = guided_point + noise
                
                # 更新当前点
                current_point = guided_point.unsqueeze(0).unsqueeze(0)
                trajectory.append(guided_point.cpu().numpy())
                time_deltas.append(time_delta)
                
                # 检查是否接近目标
                distance = torch.norm(target_point - guided_point)
                min_distance = min(min_distance, distance.item())
                
                # 如果足够接近目标，停止生成
                if distance < 0.01:  # 归一化坐标下的阈值
                    break
                
                # 如果距离不再减小，也停止
                if step > 10 and distance > min_distance * 1.5:
                    break
        
        total_time = sum(time_deltas)
        return np.array(trajectory), np.array(time_deltas), total_time

