# TouhouAI - 东方永夜抄AI自主通关项目

基于强化学习(PPO)和计算机视觉(YOLOv8)的东方永夜抄自动通关AI。

## 前言

本项目采用纯视觉来处理弹幕关系与游戏导航构建，核心为YoloV8模型+PPO强化学习+路径规划，项目目前只跑了10万步左右，可以实现简单弹幕的自动躲避，但由于作者学业及硬件能力无法跑至500万至1000万步验证最终效果，敬请见谅。
训练Yolo模型时可以使用x-anylabeling开源工具，将训练后的模型再次运用于标注中形成闭环训练，可以极大减轻标注海量弹幕数据集工作。
训练强化学习模型时请将自己训练的onnx文件或pt文件放置与同文件夹下，开始训练时可先注释掉Voronoi图寻找最优空旷点引导奖励函数，使模型专注于势场避弹训练，后期可加回来提高模型规划路径及限制行动空间。
游戏资源需自行添加在文件夹下，使用train_v2开始训练，项目可使用TensorBoard展现训练成果
作者目前大二，后续应该不会更新了，如果为你的项目起到了帮助将不胜荣幸，谢谢。

### 项目原理


### 1. 视觉感知 (YOLOv8)

使用YOLOv8目标检测模型识别游戏画面中的关键元素：

- **自机(Player)**: 玩家控制角色位置
- **弹幕(Bullets)**:
  - 小弹(bullet_small): 判定半径5px
  - 中弹(bullet_mid): 判定半径9px
  - 大弹(bullet_heavy): 判定半径24px
- **敌怪(Enemy)**: BOSS和杂兵位置
- **游戏状态**: 标题、对话、游戏中、死亡、结算等
<img width="321" height="220" alt="demo1(1)" src="https://github.com/user-attachments/assets/142c234c-1eae-4f48-b3df-4c3522fbe618" />
<img width="150" height="103" alt="demo2" src="https://github.com/user-attachments/assets/92fb58fc-c9a6-4205-8d87-5497172daf92" />
<img width="149" height="103" alt="demo3" src="https://github.com/user-attachments/assets/4dd275c3-bbfe-4894-93f5-6bff4a24b6bd" />
<img width="149" height="102" alt="demo4" src="https://github.com/user-attachments/assets/19b409cd-bb26-4c67-af31-b0997530c127" />





### 2. 弹幕预判

弹幕预判通过IOU跟踪算法和匀速外推实现。

#### 2.1 IOU跟踪算法

```
当前帧弹幕框 ←→ 上一帧弹幕框（IOU匹配）
```

- **IOU匹配**: 计算当前帧与历史帧弹幕框的交并比，IOU ≥ 0.3 视为同一颗弹幕
- **轨迹记录**: 记录每颗弹幕最近3帧的位置历史
- **稳定判定**: 连续匹配3帧以上的弹幕用于预判

#### 2.2 直行预判 (P1-P5)

基于最近2帧速度匀速外推：

```
P1位置 = 当前帧位置 + 速度 × 1
P2位置 = 当前帧位置 + 速度 × 2
P3位置 = 当前帧位置 + 速度 × 3
P4位置 = 当前帧位置 + 速度 × 4
P5位置 = 当前帧位置 + 速度 × 5
```

#### 2.3 斜向预判 (Pl1-Pl3, Pr1-Pr3) *[训练过程可选择性添加]*

预测弹幕可能扩散到的两侧区域（±45度方向）：

- **左45度旋转**: 将速度向量旋转-45度后外推
- **右45度旋转**: 将速度向量旋转+45度后外推

### 3. 人工势场(APF)避障

```python
F = k * (1/d - 1/d0) / d²
```

- `d`: 自机到弹幕中心的欧氏距离
- `d0`: 势场有效半径 (240px)
- `k`: 斥力系数 (2000000)
- 合力矢量模长 Fmag 用于避障决策和Shift切换

**Shift自动切换**:
- Fmag > 1.0 时开启低速模式
- Fmag < 0.2 时关闭低速模式

### 4. Voronoi空旷点引导 *[训练过程可选择性添加]*

通过网格搜索在游戏下半区域找到距离所有弹幕最远的安全位置。

#### 引导奖励

```python
L = 玩家到空旷点的距离
V = 0.00025 × √(810000 - L²)
```

### 5. 强化学习(PPO)

**状态空间**: 自机位置、弹幕距离、预判位置等

**动作空间**: 连续动作 [dx, dy] ∈ [-1, 1]，控制移动方向

**奖励函数**:

| 奖励项 | 值 | 说明 |
|--------|-----|------|
| 生存奖励 | +0.1/帧 | 鼓励存活 |
| 势场惩罚 | -Fmag/帧 | 远离弹幕 |
| 敌怪惩罚 | -500000/de²/帧 | 远离敌怪 |
| 生命损失 | -100 | 失去一条命 |
| 空旷点引导 | +V/帧 | *[可选择性添加]* |

## 工作流程

```
1. 游戏启动 → 检测标题画面
2. 自动导航 → 跳过对话、选择难度
3. 开始游戏 → 持续循环:
   ├── YOLO检测 → 获取自机、弹幕位置
   ├── 弹幕预判 → 计算未来位置
   ├── APF势场 → 计算避障方向
   ├── PPO决策 → 输出移动动作
   ├── 键盘控制 → 执行动作
   └── 检测死亡 → 处理复活/结束
4. 死亡处理 → 自动返回标题 → 重新开始
```

## 文件结构

```
PPO/
├── __init__.py          # 包初始化
├── touhou_env.py        # Gymnasium环境封装
├── yolo_processor.py    # YOLO检测和APF计算
├── game_navigator.py    # 游戏自动导航
├── game_manager.py      # 游戏窗口管理
├── keyboard_controller.py # 键盘控制
├── train_v2.py          # PPO训练脚本
└── requirements.txt     # 依赖列表
```

## 环境配置

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 (必须，依赖Windows API控制键盘) |
| Python | 3.8 - 3.11 |
| GPU | 推荐 NVIDIA显卡 (CUDA 11.8+)，CPU也可运行但较慢 |
| 内存 | 8GB+ |
| 游戏 | 东方永夜抄 (th08)，需单独获取 |

### Python环境创建

推荐使用Anaconda创建独立环境：

```bash
# 创建环境
conda create -n touhouAI python=3.10
conda activate touhouAI

# 或使用venv
python -m venv touhouAI_env
touhouAI_env\Scripts\activate  # Windows
```

### 依赖库安装

```bash
pip install -r PPO/requirements.txt
```

**依赖库说明**：

| 库名 | 版本 | 用途 |
|------|------|------|
| ultralytics | ≥8.0.0 | YOLOv8目标检测模型 |
| opencv-python | ≥4.0.0 | 图像处理、截图 |
| numpy | ≥1.24.0 | 数值计算 |
| torch | ≥2.0.0 | PyTorch深度学习框架 |
| torchvision | ≥0.15.0 | PyTorch视觉工具 |
| stable-baselines3 | ≥2.0.0 | PPO强化学习算法实现 |
| gymnasium | ≥1.0.0 | 强化学习环境接口 |
| tensorboard | ≥2.14.0 | 训练可视化 |
| pydirectinput | ≥1.0.0 | DirectX输入模拟 |
| pynput | ≥1.7.0 | 键盘监听 |
| psutil | ≥5.9.0 | 进程管理 |
| pywin32 | ≥305 | Windows API调用 |
| pyautogui | ≥0.9.0 | 窗口操作 |
| matplotlib | ≥3.5.0 | 数据可视化 |
| tqdm | ≥4.65.0 | 进度条 |

### GPU加速配置 (可选)

如有NVIDIA显卡，安装CUDA版本的PyTorch：

```bash
# 查看CUDA版本
nvidia-smi

# 安装对应CUDA版本的PyTorch (示例：CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 验证安装

```bash
# 验证PyTorch
python -c "import torch; print(torch.__version__)"

# 验证CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 验证YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"

# 验证stable-baselines3
python -c "from stable_baselines3 import PPO; print('PPO OK')"
```

### 常见问题

**1. pywin32安装失败**
```bash
pip install pywin32 --no-cache-dir
```

**2. torch安装慢**
使用国内镜像：
```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**3. CUDA版本不匹配**
查看当前CUDA版本后安装对应PyTorch：
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 启动训练

```bash
cd PPO
python train_v2.py
```

### 启动测试

```bash
cd PPO
python -c "from touhou_env import TouhouEnv; env = TouhouEnv(render_mode='human'); env.reset()"
```

## 配置说明

主要参数位于 `PPO/yolo_processor.py`:

- `APF_RADIUS = 240.0`: 势场有效半径
- `APF_GAIN = 2000000.0`: 斥力系数
- `SHIFT_ON_THRESHOLD = 1.0`: Shift开启阈值
- `SHIFT_OFF_THRESHOLD = 0.2`: Shift关闭阈值

## 注意事项

1. 游戏本体需单独获取，不包含在仓库中
2. 首次运行需要校准游戏窗口位置
3. YOLO模型(`.onnx`)需要在models目录下
4. 训练需要稳定的游戏运行环境
