# ECG Wearable Benchmark Beginner Guide

这份文档写给第一次接触 ECG、导联、医学信号的人。你不需要先懂心电图，也不需要先懂医学诊断。先把这件事看成一个时序信号深度学习项目：

```text
一条 ECG 波形 -> 模型 -> 5 个诊断类别的概率
```

等这条主线顺了，再慢慢补医学含义和论文创新。

## 0. 先看结论

这个项目现在做的是：

```text
用 PTB-XL 公开 ECG 数据集
训练一个小型 Transformer
判断一条 10 秒 12 导联心电图是否属于 5 类诊断
并模拟可穿戴场景中的缺导联和噪声
```

你已经跑过的 500 条数据实验，本质上是一个开发版 benchmark。它是真训练，但不是论文级完整训练。

## 1. ECG 是什么

ECG 是 electrocardiogram，中文叫心电图。

心脏每跳一次，都会产生电活动。ECG 设备把这个电活动记录成一条随时间变化的曲线。所以 ECG 不是图片，也不是表格，而是时序信号。

可以粗略理解为：

```text
时间:    0s  0.01s  0.02s  ...  10s
电压:   一串连续变化的数字
```

在项目里，PTB-XL 使用 100Hz 版本，也就是每秒采样 100 次。一条 ECG 长 10 秒，所以每条导联有：

```text
100 次/秒 × 10 秒 = 1000 个采样点
```

因此代码里常见：

```text
x = (N, 12, 1000)
```

意思是：

- `N`：有多少条 ECG 样本
- `12`：每条 ECG 有 12 个导联
- `1000`：每个导联有 1000 个时间点

## 2. 一个心跳长什么样

一段典型 ECG 里，一个心跳大概有这些结构：

```text
P 波 -> QRS 波群 -> T 波
```

简单理解：

- `P 波`：心房电活动
- `QRS 波群`：心室快速电活动，通常最尖、最明显
- `T 波`：心室恢复过程

入门阶段不用背医学定义。你只要先记住：模型看到的不是抽象标签，而是一段段有形状、有节律的波形。

建议先打开你本地跑出来的图：

```text
ecg-wearable-fm-benchmark/runs/20260626-155211/sample_ecg.png
```

如果这个目录不存在，就打开最新的：

```text
ecg-wearable-fm-benchmark/runs/<最新时间戳>/sample_ecg.png
```

## 3. 导联是什么

导联可以先理解为：

```text
从不同角度观察同一个心脏电活动
```

12 导联不是 12 个病人，也不是 12 条互不相关的数据。它们是同一个人的同一段 10 秒心电信号，只是观察角度不同。

项目里 12 个导联是：

```text
I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
```

为什么这对你的博士方向重要？

临床 ECG 通常是 12 导联，但可穿戴设备往往只有 1 到 3 个导联。比如手表、贴片、胸带，不可能像医院机器一样完整摆 10 个电极。

所以这个项目的论文主线不是简单分类，而是：

```text
临床 12 导联模型
迁移到
可穿戴少导联、有噪声、小样本 ECG 场景
还能不能稳
```

这和“可穿戴生命体征检测系统”非常贴。

## 4. PTB-XL 数据集是什么

PTB-XL 是一个公开心电数据集。它包含两万多条临床 12 导联 ECG。

本项目用的是 100Hz 低采样率版本：

```text
每条 ECG = 10 秒 × 100Hz × 12 导联
```

也就是：

```text
每条 ECG = 12 × 1000 个数字
```

项目不是预测一个单一类别，而是预测 5 个诊断大类：

```text
NORM, MI, STTC, CD, HYP
```

先按这个方式理解：

- `NORM`：正常
- `MI`：心肌梗死相关
- `STTC`：ST/T 改变相关
- `CD`：传导阻滞相关
- `HYP`：心肌肥厚相关

这 5 类是多标签。一条 ECG 可以同时有多个标签，比如既有 `MI` 又有 `STTC`。

所以标签不是这样：

```text
这条 ECG 属于第 3 类
```

而是这样：

```text
NORM: 0
MI:   1
STTC: 1
CD:   0
HYP:  0
```

这就是为什么代码用多标签 loss。

## 5. 这个项目的文件分工

项目核心代码在：

```text
ecg-wearable-fm-benchmark/src/ecg_wearable_fm/
```

你可以按这个顺序读。

### 5.1 `ptbxl.py`

作用：负责数据。

它做几件事：

- 下载 `ptbxl_database.csv`
- 下载 `scp_statements.csv`
- 把原始诊断代码映射到 5 个大类
- 下载 ECG 波形文件 `.hea` 和 `.dat`
- 用 WFDB 读取波形
- 把 ECG 归一化
- 最后整理成 `x, y, folds`

最重要的输出是：

```python
x, y, folds = build_arrays(...)
```

其中：

- `x`：ECG 波形，形状是 `(N, 12, 1000)`
- `y`：5 类诊断标签，形状是 `(N, 5)`
- `folds`：PTB-XL 官方划分，用来分训练集和测试集

### 5.2 `augment.py`

作用：模拟可穿戴 ECG 的麻烦。

里面有三种增强：

```python
lead_dropout()
```

随机把一些导联置零。模拟可穿戴设备少导联。

```python
gaussian_noise()
```

加入随机噪声。模拟传感器噪声。

```python
baseline_wander()
```

加入慢慢上下漂移的低频曲线。模拟电极接触、呼吸、运动造成的基线漂移。

训练时调用：

```python
apply_train_augmentations()
```

这就是“鲁棒训练”的入口。

### 5.3 `model.py`

作用：定义模型。

当前模型叫：

```python
PatchTransformerECG
```

它不是 ECG-FM，而是一个小型 Transformer baseline。作用是让我们先把 benchmark 跑通。

它的思路是：

```text
12 导联 ECG
-> 切成 40 个小片段
-> Transformer 编码这些片段之间的关系
-> 输出 5 个诊断类别的分数
```

为什么是 40 个片段？

默认每条 ECG 长 1000 点，`patch_size=25`：

```text
1000 / 25 = 40
```

所以模型不是一个点一个点看，而是一小段一小段看。

### 5.4 `train.py`

作用：主入口。

你运行的命令：

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train
```

就是在运行这个文件。

它负责把所有环节串起来：

```text
读取参数
-> 准备数据
-> 建 DataLoader
-> 建模型
-> 建 loss 和 optimizer
-> 每个 epoch 训练
-> 测试
-> 保存结果
```

## 6. 一次训练日志怎么读

你跑出的日志类似：

```text
Loaded PTB-XL subset: x=(500, 12, 1000), label_counts={'NORM': 111, 'MI': 198, 'STTC': 197, 'CD': 185, 'HYP': 154}
Train=452 Test=48 Device=cpu
epoch=01 train_loss=0.8995 test_loss=0.8356 macro_auroc=0.7284 macro_f1=0.564
...
epoch=10 train_loss=0.6110 test_loss=0.6902 macro_auroc=0.8108 macro_f1=0.686
```

逐句解释。

```text
x=(500, 12, 1000)
```

你用了 500 条 ECG，每条 12 导联，每个导联 1000 个采样点。

```text
label_counts
```

这表示 500 条 ECG 里每类标签出现多少次。因为是多标签，总数可以超过 500。

```text
Train=452 Test=48
```

452 条训练，48 条测试。

```text
Device=cpu
```

这次用 CPU 跑，没有用 GPU。

```text
train_loss
```

训练集上的损失。通常下降说明模型在学习。

```text
test_loss
```

测试集上的损失。它下降说明泛化可能变好，但不一定每轮都降。

```text
macro_auroc
```

越接近 1 越好，0.5 接近随机猜。这个是医学分类里很常用的指标。

```text
macro_f1
```

综合 precision 和 recall，反映预测标签是否命中。

你这次 500 条实验从：

```text
macro_auroc=0.7284
```

到：

```text
macro_auroc=0.8108
```

说明模型确实学到了东西。

## 7. 为什么用 `BCEWithLogitsLoss`

普通单标签分类常用：

```python
CrossEntropyLoss
```

它适合这种问题：

```text
猫 / 狗 / 鸟，只能选一个
```

但 ECG 诊断是多标签：

```text
一条 ECG 可以既有 MI，也有 STTC
```

所以代码用：

```python
nn.BCEWithLogitsLoss()
```

它会把 5 个类别分别当成 5 个“是/否”问题：

```text
是不是 NORM？
是不是 MI？
是不是 STTC？
是不是 CD？
是不是 HYP？
```

## 8. 为什么要做增强

如果只在干净 12 导联临床 ECG 上训练，模型可能很依赖完整导联和干净波形。

但可穿戴场景里常见问题是：

- 导联少
- 电极接触不稳定
- 运动噪声
- 基线漂移
- 标签少

所以本项目用增强模拟这些问题。

命令里的：

```powershell
--lead-dropout 0.25
```

表示训练时随机丢掉一部分导联。

```powershell
--noise-std 0.02
```

表示加入随机噪声。

```powershell
--baseline-wander 0.03
```

表示加入低频漂移。

这就是“可穿戴鲁棒性”的第一步。

## 9. 入门练习

### 练习 1：看图识数据

打开：

```text
ecg-wearable-fm-benchmark/runs/<时间戳>/sample_ecg.png
```

观察：

- 12 行是不是 12 个导联
- I、II、V1、V6 的波形是不是不一样
- 每隔一段时间是不是有尖尖的 QRS 波

目标：先把 ECG 当成真实波形，不要只当成数组。

### 练习 2：跑干净 baseline

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 500 `
  --epochs 10 `
  --batch-size 32 `
  --d-model 96 `
  --layers 2 `
  --heads 4 `
  --lead-dropout 0 `
  --noise-std 0 `
  --baseline-wander 0
```

这是不加增强的版本。

### 练习 3：跑鲁棒 baseline

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 500 `
  --epochs 10 `
  --batch-size 32 `
  --d-model 96 `
  --layers 2 `
  --heads 4 `
  --lead-dropout 0.25 `
  --noise-std 0.02 `
  --baseline-wander 0.03
```

这是你已经跑过的方向。

比较两个 run 的：

```text
metrics.json -> final -> macro_auroc
metrics.json -> final -> macro_f1
```

### 练习 4：测试少导联

3 导联：

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 500 `
  --epochs 10 `
  --batch-size 32 `
  --d-model 96 `
  --layers 2 `
  --heads 4 `
  --eval-leads I II V1
```

单导联：

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 500 `
  --epochs 10 `
  --batch-size 32 `
  --d-model 96 `
  --layers 2 `
  --heads 4 `
  --eval-leads I
```

注意：当前实现里 `--eval-leads` 会在测试时只保留指定导联。训练仍按命令里的增强设置进行。

## 10. 如何从入门实验走向论文

现在这个项目还不是论文，只是一个可靠起点。

论文路线可以这样长出来：

```text
第一步：PTB-XL baseline 跑通
第二步：比较 clean vs robust augmentation
第三步：测试 12 导联、3 导联、1 导联性能下降
第四步：接入 ECG-FM 或 ECGFounder
第五步：提出自己的鲁棒适配方法
第六步：扩展到外部数据集验证
```

真正的论文问题不是：

```text
我能不能把 PTB-XL 分数刷高？
```

而是：

```text
一个在临床 12 导联 ECG 上训练得很好的模型，
到了可穿戴少导联、噪声、小样本场景，
为什么会掉性能？
怎样用轻量方法让它更稳？
```

这才和可穿戴生命体征、医学人工智能、博士研究方向接得上。

## 11. 你学完后应该能回答

- ECG 为什么是时序信号？
- 12 导联是什么意思？
- 为什么输入是 `(N, 12, 1000)`？
- PTB-XL 的 5 类标签分别代表什么？
- `ptbxl.py`、`augment.py`、`model.py`、`train.py` 各自负责什么？
- 为什么这个任务用 `BCEWithLogitsLoss`？
- `macro_auroc` 和 `macro_f1` 大概怎么看？
- 为什么少导联和噪声是可穿戴 ECG 的关键问题？
- 这个项目如何从入门实验长成论文方向？

## 12. 最小心智模型

如果只记一版，就记这个：

```text
ECG 是心脏电活动的时间序列。
12 导联是 12 个观察角度。
PTB-XL 每条样本是 12 × 1000 的数字矩阵。
模型输入这个矩阵，输出 5 个诊断概率。
本项目关心模型在少导联和噪声下是否仍然可靠。
```
