# LSTM鼠标轨迹生成项目

这是一个使用LSTM（长短期记忆网络）模型来生成模拟人类鼠标移动轨迹的项目。给定起点和目标点，模型可以生成自然、流畅的鼠标移动轨迹。

## 项目特点

- 🎯 **智能轨迹生成**: 使用LSTM模型学习人类鼠标移动模式
- 🖱️ **自然移动**: 生成的轨迹模拟真实人类鼠标移动的不规则性和曲线特征
- 📊 **可视化展示**: 提供轨迹可视化功能，直观展示生成结果
- 🔧 **易于使用**: 简单的API接口，方便集成到其他项目

## 项目结构

```
mouse_trajectory_lstm/
├── data_generator.py      # 数据生成脚本（模拟数据）
├── collect_trajectory.py  # 鼠标轨迹收集脚本（真实数据）
├── visualize_collected.py # 可视化收集的轨迹数据
├── model.py              # LSTM模型定义
├── train.py              # 模型训练脚本
├── generate.py           # 轨迹生成脚本
├── example.py            # 使用示例
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- pynput (用于收集真实鼠标轨迹)

## 安装步骤

1. **安装依赖**

```bash
pip install -r requirements.txt
```

2. **生成训练数据**

有两种方式生成训练数据：

**方式1: 使用模拟数据生成器（快速）**

```bash
cd mouse_trajectory_lstm
python data_generator.py
```

这将生成 `trajectories.npy` 和 `targets.npy` 文件，包含模拟的训练数据。

**方式2: 收集真实鼠标轨迹（推荐，质量更高）**

```bash
python collect_trajectory.py
```

运行后会开始监听鼠标移动，正常使用鼠标即可收集轨迹数据。收集的数据会保存在 `collected_data/` 目录下。

**参数说明：**
- `--screen-width`: 屏幕宽度（像素），默认自动检测
- `--screen-height`: 屏幕高度（像素），默认自动检测
- `--min-length`: 最小轨迹长度（点数），默认10
- `--max-idle`: 最大空闲时间（秒），默认2.0
- `--min-distance`: 最小移动距离（像素），默认10
- `--merge`: 合并已收集的数据文件

**合并多个收集的数据文件：**

```bash
python collect_trajectory.py --merge
```

合并后的数据会保存为 `trajectories_merged.npy` 和 `targets_merged.npy`。

**可视化收集的数据：**

```bash
python visualize_collected.py
```

这将显示所有收集到的鼠标轨迹，包括：
- 所有轨迹的概览图
- 每条轨迹的起点和终点标记
- 轨迹点的分布
- 统计信息（轨迹数量、平均长度、效率等）

**可视化参数说明：**
- `--data-dir`: 数据目录，默认 `collected_data`
- `--timestamp`: 指定要可视化的数据时间戳（默认使用最新的）
- `--mode`: 可视化模式
  - `all`: 显示所有轨迹的概览图
  - `detail`: 详细显示指定轨迹
  - `both`: 同时显示概览和详细图
- `--max-trajectories`: 最多显示的轨迹数（用于all模式）
- `--trajectory-indices`: 要详细显示的轨迹索引（用于detail模式）
- `--save-all`: 保存概览图的路径
- `--save-detail`: 保存详细图的路径

**示例：**
```bash
# 显示所有轨迹
python visualize_collected.py --mode all

# 详细显示前4条轨迹
python visualize_collected.py --mode detail

# 显示指定索引的轨迹
python visualize_collected.py --mode detail --trajectory-indices 0 1 2 3

# 保存图表
python visualize_collected.py --mode both --save-all overview.png --save-detail detail.png
```

3. **训练模型**

```bash
python train.py
```

训练过程会自动保存最佳模型到 `mouse_trajectory_model.pth`。

4. **生成轨迹**

```bash
python generate.py
```

这将使用训练好的模型生成几个示例轨迹，并保存可视化图片。

## 使用方法

### 基本使用

```python
from model import MouseTrajectoryLSTM
import torch
import numpy as np

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MouseTrajectoryLSTM(input_size=2, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load('mouse_trajectory_model.pth', map_location=device))
model.eval()

# 定义起点和目标点（归一化坐标，范围[0, 1]）
start_point = np.array([0.1, 0.1])  # 起点
target_point = np.array([0.9, 0.9])  # 目标点

# 生成轨迹
trajectory = model.generate_trajectory(
    start_point=start_point,
    target_point=target_point,
    max_steps=200,
    device=device
)

# 轨迹点已归一化，需要乘以屏幕尺寸得到实际坐标
screen_width, screen_height = 1920, 1080
trajectory_screen = trajectory * np.array([screen_width, screen_height])
```

### 自定义参数

在 `generate.py` 中可以修改以下参数：

- `screen_width`, `screen_height`: 屏幕尺寸
- `max_steps`: 生成轨迹的最大步数
- `temperature`: 控制生成轨迹的随机性

## 模型架构

- **输入**: 鼠标轨迹点序列 (x, y坐标)
- **LSTM层**: 2层LSTM，隐藏层大小128
- **输出层**: 全连接层，输出下一个轨迹点的坐标
- **损失函数**: 均方误差 (MSE)

## 训练参数

- **批次大小**: 32
- **学习率**: 0.001 (使用Adam优化器)
- **训练轮数**: 50
- **学习率调度**: ReduceLROnPlateau（当验证损失不再下降时降低学习率）

## 数据生成

### 模拟数据生成

`data_generator.py` 使用以下方法生成模拟人类鼠标移动的数据：

1. **贝塞尔曲线**: 生成平滑的曲线路径
2. **随机扰动**: 添加小幅度抖动模拟手部微颤
3. **速度变化**: 模拟人类移动时开始和结束较慢、中间较快的特征

### 真实数据收集

`collect_trajectory.py` 可以实时收集用户的真实鼠标移动轨迹：

1. **自动检测屏幕尺寸**: 自动获取当前屏幕分辨率
2. **智能轨迹分割**: 
   - 鼠标静止超过设定时间（默认2秒）自动结束当前轨迹
   - 鼠标点击也会结束当前轨迹并开始新轨迹
3. **数据过滤**: 自动过滤微小抖动，只记录有意义的移动
4. **数据归一化**: 自动将坐标归一化到[0,1]范围，与训练数据格式兼容
5. **批量收集**: 支持多次收集后合并数据

**使用建议：**
- 正常使用鼠标进行各种操作（点击、拖拽、移动等）
- 收集足够多的轨迹（建议至少1000条以上）以获得更好的训练效果
- 可以分多次收集，然后使用 `--merge` 参数合并数据

## 性能优化

- 支持GPU加速训练（如果可用）
- 使用梯度裁剪防止梯度爆炸
- Dropout防止过拟合
- 学习率自适应调整

## 注意事项

1. 首次运行需要生成训练数据，可能需要几分钟时间
2. 训练过程可能需要较长时间，建议使用GPU加速
3. 生成的轨迹坐标是归一化的（0-1范围），使用时需要乘以屏幕尺寸

## 未来改进方向

- [x] 支持收集真实鼠标轨迹数据进行训练
- [ ] 添加更多特征（如移动速度、加速度）
- [ ] 支持不同移动风格（快速、慢速、精确等）
- [ ] 优化模型架构，提高生成质量
- [ ] 添加轨迹平滑和优化功能
- [ ] 支持实时轨迹收集和训练

## 许可证

本项目仅供学习和研究使用。

## 作者

AI Assistant

