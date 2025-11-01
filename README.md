# 🐱🐶 基于 CNN 的猫狗分类系统

> 基于 TensorFlow 实现的猫狗二分类项目，包含数据预处理、模型搭建、训练与推理全流程。  
> 通过深度学习模型识别输入图片是“猫”还是“狗”，并输出类别与置信度。

<p align="left">
  <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Nuyoahwjl/Cats-Vs-Dogs?style=social">
  <img alt="Top Language" src="https://img.shields.io/github/languages/top/Nuyoahwjl/Cats-Vs-Dogs?logo=python&logoColor=white&color=3776AB">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x%20%7C%201.15-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/Nuyoahwjl/Cats-Vs-Dogs">
</p>

---

## ✨ 功能特性

- 🧠 使用卷积神经网络（CNN）进行猫/狗二分类
- 🧰 完整的训练、评估、推理脚本
- 🗂️ 简洁的项目结构与可复用的数据管道
- 📈 提供训练过程可视化与预测示例
- ⚙️ 支持 TensorFlow 2.x（兼容模式）与 1.15（传统环境）

---

## 📁 目录结构

```
cats_vs_dogs/
├── data/                 
│   ├── train/
│   │   ├── cat.1.jpg
│   │   ├── dog.1.jpg
│   │   └── ...
│   └── test/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── image/                 # 训练过程图 / 预测结果图
├── log/                   # 训练的模型与参数（学习率 0.0001）
├── res/                   # classify.py 的分类输出结果
│
├── input_data.py          # 数据预处理：加载图片与标签、批处理、标准化
├── model.py               # 定义 CNN 模型、loss、optimizer、评价指标
├── training.py            # 训练循环与模型保存
├── test.py                # 单张随机测试与可视化
├── classify.py            # 批量预测与分类（示例为 200 张）
│
└── README.md
```

---

## 🧩 环境要求

- Python 3.12
- TensorFlow
  - 推荐：TensorFlow 2.x（使用 `tensorflow.compat.v1` 兼容模式）
  - 备选：TensorFlow 1.15（更传统，但对 Python 版本有约束）
- 依赖：NumPy、Matplotlib、Pillow

> 若使用 TF 2.x，请确保启用 v1 兼容模式：
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

---

## 📦 安装依赖

- 方案 A（推荐，TF 2.x 兼容模式）：
```bash
pip install "tensorflow>=2.10,<3" numpy matplotlib pillow
```

- 方案 B（传统环境，TF 1.15）：
```bash
pip install tensorflow==1.15 numpy matplotlib pillow
```

---

## 🚀 快速开始

### 1) 准备数据集
将数据集按如下目录组织：
```
data/
├── train/
│   ├── cat.1.jpg
│   ├── dog.1.jpg
│   └── ...
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```
- `train/` 中包含标注好的猫/狗图片（可按文件名或子目录组织）。
- `test/` 用于推理演示与可视化。

### 2) 训练模型
可在 `training.py` 中调整关键超参数：
```python
BATCH_SIZE = 16  # 每批次读取数据的数量
CAPACITY = 2000  # 队列最大容量
MAX_STEP = 10000  # 训练最大步数，一般5K~10k
learning_rate = 0.0001  # 学习率，一般小于0.0001
```
启动训练：
```bash
python training.py
```
- 每 50 步打印一次 loss 与 acc
- 每 100 步记录并绘图
- 每 5000 步保存一次模型

### 3) 测试与预测
随机选择一张 `data/test/` 中的图片进行预测并可视化：
```bash
python test.py
```

### 4) 批量分类（示例：200 张）
将批量预测结果输出至 `res/`：
```bash
python classify.py
```

---

## 🧠 关键模块说明

- [`input_data.py`](./input_data.py)
  - `get_files`：读取训练集图片路径并生成标签
  - `get_batch`：批量加载、缩放与标准化
- [`model.py`](./model.py)
  - `cnn_inference`：CNN 网络结构（卷积/池化/全连接）
  - `losses`：交叉熵损失
  - `training`：Adam 优化器最小化损失
  - `evaluation`：计算准确率
- [`training.py`](./training.py)
  - 训练循环、日志记录、模型保存
- [`test.py`](./test.py)
  - 加载模型，随机选取图片预测与可视化
- [`classify.py`](./classify.py)
  - 批量预测，并将结果保存到 `res/`

---

## 📊 训练可视化与预测示例

- 训练过程（Accuracy & Loss）
  
  ![训练曲线](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Accuracy&Loss.png)

- 预测结果示例

  ![预测示例](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Prediction.png)
  
> [!important]
> 说明：训练曲线每 100 步记录一次，未进行平滑处理，曲线较为“锯齿”属正常现象。

---

## ⚠️ 注意事项

- 确保数据集路径配置正确，且图片格式一致
- 使用 TensorFlow 2.x 时需启用 v1 兼容模式（见上文）
- 若使用 TensorFlow 1.15，请注意其与 Python 高版本的兼容性

---

## 🙌 参与与支持

- 如果这个项目对你有帮助，欢迎点亮 ⭐ Star 支持！
- 欢迎提交 Issue 或 PR 一起改进项目～

