# 执行日志

## 2026-06-26

- 创建博士执行计划工作区。
- 将原始规划转化为可维护的 `task_plan.md`，包含阶段、验收标准、近期清单、风险与完成定义。
- 创建 `findings.md`，沉淀公开资料判断、来源、博士定位推断和待确认问题。
- 创建 `progress.md`，用于记录后续推进。
- 下一步：按 `roadmap_36_months.md` 和 `weekly_execution_template.md` 开始每周执行。

## 2026-06-26 ECG 深度学习 baseline

- 创建 `ecg-dl-baseline/` 小型开源项目骨架。
- 建立 `.venv` 并安装 PyTorch CPU、WFDB、scikit-learn、matplotlib 等依赖。
- 实现 MIT-BIH 直连下载、AAMI 风格心拍粗分类、1D-CNN 训练、指标导出和图像保存。
- 跑通 smoke run：MIT-BIH 100/101/200/207/208，1953 个心拍，2 个 epoch，accuracy=0.8875，balanced_accuracy=0.5821。
- 发现后续重点：记录级验证、类别不均衡处理、S/Q 类别样本不足、CWT/STFT 与 TCN/Transformer baseline。

## 2026-06-26 ECG foundation/可穿戴鲁棒 benchmark

- 创建 `ecg-wearable-fm-benchmark/`，作为 PTB-XL + foundation-model 迁移 + 可穿戴鲁棒性的论文主线项目。
- 实现 PTB-XL metadata 下载、诊断 superclass 标签解析、100Hz 波形直连下载和缓存。
- 实现 lead dropout、Gaussian noise、baseline wander 三类可穿戴场景增强。
- 实现 compact patch Transformer 多标签分类 baseline，指标包含 macro AUROC、macro AUPRC、macro F1 和逐类结果。
- 跑通 smoke run：40 条 PTB-XL，1 epoch，CPU，macro AUROC=0.7256，macro AUPRC=0.7249，macro F1=0.4691。
- 优化 waveform 并行下载，后续可扩大到 500/全量 PTB-XL。

## 2026-06-26 ECG 零基础讲解文档

- 新增 `ecg-wearable-fm-benchmark/BEGINNER_GUIDE.md`。
- 按“ECG 是什么 -> 导联是什么 -> PTB-XL 数据 -> 代码文件分工 -> 训练日志 -> 入门练习 -> 论文路线”的顺序组织。
- README 增加零基础入口，避免初学者直接被 ECG-FM、导联、AUROC、Transformer 等术语淹没。

## 2026-06-26 GitHub 上传准备

- 确认 GitHub CLI 已登录 `pengliu0804`。
- 确认目标仓库 `pengliu0804/lw` 尚不存在，准备新建。
- 新增根目录 `.gitignore`，排除 `.venv/`、数据目录、训练输出、模型权重和缓存。
- 新增根目录 `README.md`，说明规划文件和两个 ECG starter project 的用途。
- 初始化 Git 仓库，创建公开远程仓库 `https://github.com/pengliu0804/lw`，并推送 `main` 分支。
