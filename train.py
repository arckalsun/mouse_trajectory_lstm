"""
训练LSTM鼠标轨迹生成模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from model import MouseTrajectoryLSTM


class TrajectoryDataset(Dataset):
    """鼠标轨迹数据集"""
    
    def __init__(self, trajectories, targets, time_intervals=None):
        """
        Args:
            trajectories: 轨迹序列 (num_samples, seq_len, 2)
            targets: 目标点 (num_samples, 2) 或 (num_samples, 3) [x, y, time]
            time_intervals: 时间间隔数组 (num_samples, seq_len-1)，可选
        """
        self.trajectories = trajectories
        self.targets = targets
        self.time_intervals = time_intervals
        
        # 如果没有提供时间间隔，从轨迹中计算（假设均匀采样）
        if time_intervals is None:
            # 从targets中提取时间信息（如果存在）
            if targets.shape[1] >= 3:
                # targets包含时间信息，提取最后一个点的时间间隔
                self.time_intervals = None  # 需要从轨迹序列中计算
            else:
                # 使用默认时间间隔（估算）
                self.time_intervals = None
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = torch.FloatTensor(self.trajectories[idx])
        target = torch.FloatTensor(self.targets[idx])
        
        # 如果target包含时间信息，提取坐标和时间
        if target.shape[0] >= 3:
            target_coord = target[:2]  # 坐标
            target_time = target[2] if target.shape[0] > 2 else 0.01  # 时间间隔
            # 构建包含坐标和时间的target
            target_with_time = torch.cat([target_coord, torch.tensor([target_time])])
        else:
            target_coord = target
            target_time = 0.01  # 默认时间间隔
            target_with_time = torch.cat([target_coord, torch.tensor([target_time])])
        
        return trajectory, target_with_time


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cpu', save_path='model.pth'):
    """
    训练模型
    
    Args:
        model: LSTM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        save_path: 模型保存路径
    """
    model = model.to(device)
    # 使用组合损失：坐标损失 + 时间损失
    coord_criterion = nn.MSELoss()
    time_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"开始训练，设备: {device}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for trajectories, targets in train_pbar:
            trajectories = trajectories.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            # 使用轨迹序列预测目标点和时间间隔
            predictions, _ = model(trajectories)
            
            # 分离坐标和时间间隔
            if predictions.shape[1] == 3 and targets.shape[1] == 3:
                # 模型输出包含时间间隔
                pred_coords = predictions[:, :2]
                pred_time = predictions[:, 2]
                target_coords = targets[:, :2]
                target_time = targets[:, 2]
                
                # 计算坐标损失和时间损失
                coord_loss = coord_criterion(pred_coords, target_coords)
                time_loss = time_criterion(pred_time, target_time)
                
                # 组合损失（坐标损失权重更高）
                loss = coord_loss + 0.1 * time_loss
            else:
                # 兼容旧版本：只预测坐标
                if predictions.shape[1] == 3:
                    predictions = predictions[:, :2]
                if targets.shape[1] == 3:
                    targets = targets[:, :2]
                loss = coord_criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for trajectories, targets in val_pbar:
                trajectories = trajectories.to(device)
                targets = targets.to(device)
                
                predictions, _ = model(trajectories)
                
                # 分离坐标和时间间隔
                if predictions.shape[1] == 3 and targets.shape[1] == 3:
                    pred_coords = predictions[:, :2]
                    pred_time = predictions[:, 2]
                    target_coords = targets[:, :2]
                    target_time = targets[:, 2]
                    
                    coord_loss = coord_criterion(pred_coords, target_coords)
                    time_loss = time_criterion(pred_time, target_time)
                    loss = coord_loss + 0.1 * time_loss
                else:
                    if predictions.shape[1] == 3:
                        predictions = predictions[:, :2]
                    if targets.shape[1] == 3:
                        targets = targets[:, :2]
                    loss = coord_criterion(predictions, targets)
                
                val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, save_path)
            print(f"✓ 保存最佳模型 (验证损失: {avg_val_loss:.6f})")
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    return train_losses, val_losses


def main():
    """主训练函数"""
    # 检查数据文件
    if not os.path.exists('trained_data/trajectories.npy') or not os.path.exists('trained_data/targets.npy'):
        print("未找到训练数据文件，请先运行 data_generator.py 生成数据")
        return
    
    # 加载数据
    print("加载训练数据...")
    trajectories = np.load('trained_data/trajectories.npy')
    targets = np.load('trained_data/targets.npy')
    
    print(f"数据形状: 轨迹 {trajectories.shape}, 目标 {targets.shape}")
    
    # 划分训练集和验证集
    split_idx = int(len(trajectories) * 0.8)
    train_trajectories = trajectories[:split_idx]
    train_targets = targets[:split_idx]
    val_trajectories = trajectories[split_idx:]
    val_targets = targets[split_idx:]
    
    # 创建数据集和数据加载器
    train_dataset = TrajectoryDataset(train_trajectories, train_targets)
    val_dataset = TrajectoryDataset(val_trajectories, val_targets)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    # 创建模型（启用时间间隔输出）
    model = MouseTrajectoryLSTM(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_time=True  # 启用时间间隔输出
    )
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs("model", exist_ok=True)

    # 训练模型
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        save_path='model/mouse_trajectory_model.pth'
    )
    
    print("\n训练完成！模型已保存到 model/mouse_trajectory_model.pth")


if __name__ == "__main__":
    main()

