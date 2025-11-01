# 🐱🐶 基于 CNN 的猫狗分类系统

> 基于 TensorFlow 实现的猫狗二分类项目，包含数据预处理、模型搭建、训练与推理全流程。  
> 通过深度学习模型识别输入图片是“猫”还是“狗”，并输出类别与置信度。

<p align="left">
  <a href="https://github.com/Nuyoahwjl/Cats-Vs-Dogs/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Nuyoahwjl/Cats-Vs-Dogs?style=social">
  </a>
  <a href="https://github.com/Nuyoahwjl/Cats-Vs-Dogs">
    <img alt="Top Language" src="https://img.shields.io/github/languages/top/Nuyoahwjl/Cats-Vs-Dogs?logo=python&logoColor=white&color=3776AB">
  </a>
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
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0001
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
  
  ![训练曲线]([./image/Accuracy%26Loss.png](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Accuracy&Loss.png))

- 预测结果示例

  ![预测示例]([./image/Prediction.png](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Prediction.png))
  
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


<!--

# 基于CNN的猫狗分类系统

这是一个基于卷积神经网络（CNN）实现的猫狗分类系统。通过深度学习，模型能够识别图像中的猫或狗，输出其类别及概率。该项目主要分为数据预处理、模型构建、训练和评估四个部分。

## 目录结构

```
cats_vs_dogs/
├── data/                 
│   ├── train/
│   │   ├── cat.1.jpg
│   │   ├── dog.1.jpg
│   │   ├── ...
│   └── test/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── image/             # 训练图&结果图
├── log/               # 训练模型和参数(学习率0.0001)
├── res/               # 存放classify.py分类结果
│
├── input_data.py      # 数据预处理，加载图片及标签
├── model.py           # 定义CNN神经网络模型
├── training.py        # 模型训练和保存
├── test.py            # 模型测试与预测
├── classify.py        # 预测200张图片并分类
│
└── README.md          # 项目说明文档
```

## 环境要求

- Python 3.12.2
- TensorFlow 2.18.0 (使用 `tensorflow.compat.v1` 模式)
- NumPy
- Matplotlib
- Pillow

## 安装依赖

可以使用 `pip` 安装项目所需的依赖包：

```bash
pip install tensorflow==1.15 numpy matplotlib pillow
```

## 使用方法

### 1. 数据准备

需要准备一个猫狗数据集，该数据集应包含猫和狗的图片。

```
data/
├── train/
│   ├── cat.1.jpg
│   ├── dog.1.jpg
│   ├── ...
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

其中，`train/` 文件夹包含猫狗图片，`test/` 文件夹用于存放测试图片。标签由文件夹的名称确定（`train/cat` 表示猫，`train/dog` 表示狗）。

### 2. 数据预处理

在 `input_data.py` 文件中，`get_files` 函数会读取指定目录下的图片文件并生成标签。`get_batch` 函数则会对图片进行批量处理、缩放和标准化操作。

### 3. 训练模型

训练模型使用 `training.py` 文件。执行此脚本时，模型将根据准备好的数据进行训练，并保存模型参数。

可以修改以下训练参数：

```python
BATCH_SIZE = 16  # 每批次读取数据的数量
CAPACITY = 2000  # 队列最大容量
MAX_STEP = 10000  # 训练最大步数，一般5K~10k
learning_rate = 0.0001  # 学习率，一般小于0.0001
```

执行训练脚本：

```bash
python training.py
```

训练过程中每隔50步打印一次当前的loss以及acc。每100步记录数据并描点作图。每5000步会保存一次训练好的模型，最终保存的模型将用于测试和预测。训练结束将打印训练图。

### 4. 测试与预测

训练完成后，可以使用 `test.py` 对新图片进行分类预测。该脚本会随机从测试集目录中选择一张图片，并使用训练好的模型进行分类预测，输出图像属于猫或狗的概率。

执行测试脚本：

```bash
python test.py
```

此脚本会显示选取的测试图片，并打印出该图片是猫还是狗的预测结果以及对应的概率。

### 5. 项目中主要函数的作用

- **`input_data.py`**：
  
  - `get_files`: 获取训练集中的图像文件路径及对应标签。
  - `get_batch`: 批量加载和处理图像，进行图像的缩放和标准化。

- **`model.py`**：
  
  - `cnn_inference`: 定义卷积神经网络的结构，包含卷积层、池化层、全连接层。
  - `losses`: 计算损失函数（交叉熵）。
  - `training`: 定义训练操作，使用 Adam 优化器来最小化损失。
  - `evaluation`: 计算模型的准确率。

- **`training.py`**：
  
  - 负责模型训练，包括获取数据批次、计算损失、更新模型参数和保存模型。
  - ，每100步记录训每50步打印一次当前损失和准确率练过程中的准确率和损失值，每5000步保存一次训练模型。

- **`test.py`**：
  
  - 加载训练好的模型，并对随机选取的一张图片进行猫狗分类预测。
  - 显示预测结果以及对应的概率。

## 项目输出

- 在训练过程中，模型的损失和准确率会显示在控制台，并且每5000步保存一次模型。
- `test.py` 会输出一张测试图片以及该图片是猫还是狗的预测结果。

## 注意事项

- 确保数据集路径正确，并且训练集和测试集的图像格式一致。
- 该项目使用 TensorFlow 1.x，如果你使用的是 TensorFlow 2.x，确保启用了兼容模式：`import tensorflow.compat.v1 as tf` 并调用 `tf.disable_v2_behavior()`。

## 训练图（准确率与损失值）
> [!important]
> 由于每100步记录一次数据且没有做平滑处理，此图不太美观

![准确率和损失值图](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Accuracy&Loss.png)  

## 预测结果图

![预测结果图](https://cdn.jsdelivr.net/gh//Nuyoahwjl/Cats-Vs-Dogs/image/Prediction.png)

-->
