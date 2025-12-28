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
    """鼠标轨迹数据集（支持时间信息）"""
    
    def __init__(self, trajectories, targets, time_deltas=None):
        """
        Args:
            trajectories: 轨迹序列 (num_samples, seq_len, 2)
            targets: 目标点 (num_samples, 2) 或 (num_samples, 3) [x, y, time_delta]
            time_deltas: 时间间隔数组 (num_samples,)，可选
        """
        self.trajectories = trajectories
        self.targets = targets
        
        # 如果targets是2维，尝试从time_deltas添加时间信息
        if time_deltas is not None and targets.shape[1] == 2:
            # 将时间信息添加到targets
            targets_with_time = np.zeros((len(targets), 3), dtype=np.float32)
            targets_with_time[:, :2] = targets
            targets_with_time[:, 2] = time_deltas
            self.targets = targets_with_time
        elif targets.shape[1] == 2:
            # 没有时间信息，使用默认值0
            targets_with_time = np.zeros((len(targets), 3), dtype=np.float32)
            targets_with_time[:, :2] = targets
            targets_with_time[:, 2] = 0.0
            self.targets = targets_with_time
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = torch.FloatTensor(self.trajectories[idx])
        target = torch.FloatTensor(self.targets[idx])
        return trajectory, target


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
    criterion = nn.MSELoss()
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
            
            # 计算损失
            # 如果targets是3维（包含时间），分别计算坐标损失和时间损失
            if targets.shape[1] == 3:
                # 分离坐标和时间
                coord_targets = targets[:, :2]
                time_targets = targets[:, 2:3]
                coord_preds = predictions[:, :2]
                time_preds = predictions[:, 2:3]
                
                # 计算坐标损失和时间损失（可以设置不同的权重）
                coord_loss = criterion(coord_preds, coord_targets)
                time_loss = criterion(time_preds, time_targets)
                loss = coord_loss + 0.1 * time_loss  # 时间损失的权重较小
            else:
                # 兼容旧数据（只有坐标）
                if predictions.shape[1] == 3:
                    predictions = predictions[:, :2]
                loss = criterion(predictions, targets)
            
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
                
                # 计算损失（与训练阶段相同）
                if targets.shape[1] == 3:
                    coord_targets = targets[:, :2]
                    time_targets = targets[:, 2:3]
                    coord_preds = predictions[:, :2]
                    time_preds = predictions[:, 2:3]
                    
                    coord_loss = criterion(coord_preds, coord_targets)
                    time_loss = criterion(time_preds, time_targets)
                    loss = coord_loss + 0.1 * time_loss
                else:
                    if predictions.shape[1] == 3:
                        predictions = predictions[:, :2]
                    loss = criterion(predictions, targets)
                
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


def extract_time_deltas_from_metadata(metadata_dir="collected_data"):
    """
    从收集的数据元数据中提取时间间隔信息
    
    Args:
        metadata_dir: 元数据目录
    
    Returns:
        time_deltas: 时间间隔数组 (num_samples,)，每个样本到达目标点后的停留时间
    """
    import glob
    import json
    
    time_deltas = []
    metadata_files = glob.glob(os.path.join(metadata_dir, "metadata_*.json"))
    
    for metadata_file in sorted(metadata_files):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if 'timestamps' in metadata:
                timestamps_list = metadata['timestamps']
                for timestamps in timestamps_list:
                    if len(timestamps) > 1:
                        # 计算到达目标点后的停留时间（最后一个时间间隔）
                        # 或者使用轨迹总时间
                        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
                        # 也可以使用最后一个时间间隔
                        last_interval = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 0.0
                        time_deltas.append(max(total_time, last_interval))
                    else:
                        time_deltas.append(0.0)
        except:
            pass
    
    return np.array(time_deltas, dtype=np.float32) if time_deltas else None


def prepare_targets_with_time(targets, time_deltas=None):
    """
    准备包含时间信息的目标数组
    
    Args:
        targets: 目标点数组 (num_samples, 2)
        time_deltas: 时间间隔数组 (num_samples,)，可选
    
    Returns:
        targets_with_time: (num_samples, 3) 或 (num_samples, 2)
    """
    if time_deltas is not None and len(time_deltas) == len(targets):
        targets_with_time = np.zeros((len(targets), 3), dtype=np.float32)
        targets_with_time[:, :2] = targets
        # 归一化时间间隔（假设最大停留时间为5秒）
        max_time = max(np.max(time_deltas), 5.0)
        targets_with_time[:, 2] = np.clip(time_deltas / max_time, 0.0, 1.0)
        return targets_with_time
    else:
        # 没有时间信息，返回原始targets
        return targets


def main():
    """主训练函数"""
    # 检查数据文件
    if not os.path.exists('merged/trajectories.npy') or not os.path.exists('merged/targets.npy'):
        if not os.path.exists('trained_data/trajectories.npy') or not os.path.exists('trained_data/targets.npy'):
            print("未找到训练数据文件，请先运行 collect_trajectory.py --merge 合并数据")
            return
        else:
            trajectories_file = 'trained_data/trajectories.npy'
            targets_file = 'trained_data/targets.npy'
    else:
        trajectories_file = 'merged/trajectories.npy'
        targets_file = 'merged/targets.npy'
    
    # 加载数据
    print("加载训练数据...")
    trajectories = np.load(trajectories_file)
    targets = np.load(targets_file)
    
    print(f"数据形状: 轨迹 {trajectories.shape}, 目标 {targets.shape}")
    
    # 尝试从元数据中提取时间信息
    print("尝试从收集的数据中提取时间信息...")
    time_deltas = extract_time_deltas_from_metadata()
    
    if time_deltas is not None and len(time_deltas) == len(targets):
        print(f"成功提取时间信息: {len(time_deltas)} 个样本")
        targets = prepare_targets_with_time(targets, time_deltas)
        print(f"目标数组形状更新为: {targets.shape} (包含时间信息)")
    else:
        print("未找到时间信息，将使用默认值（时间=0）")
        # 如果没有时间信息，添加默认时间列
        if targets.shape[1] == 2:
            targets = prepare_targets_with_time(targets, None)
    
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
    
    # 创建模型（启用时间输出）
    model = MouseTrajectoryLSTM(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_time=True  # 启用时间输出
    )
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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

