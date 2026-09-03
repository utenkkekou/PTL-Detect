# PTL-Detect

**行人红绿灯视觉工具 · Pedestrian Traffic Light Toolkit**

[![Tests](https://github.com/utenkkekou/PTL-Detect/actions/workflows/tests.yml/badge.svg)](https://github.com/utenkkekou/PTL-Detect/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)

围绕行人信号灯建立的标注、数据整理、YOLO 训练和误判分析工具链。重点分析红灯误识别为绿灯、车行灯混淆、遮挡与小目标等问题，包含浏览器人工复核界面和 Core ML 导出脚本。

Annotation, dataset preparation, YOLO training and error analysis for pedestrian traffic lights, with a browser review tool and an optional Core ML export pipeline.

## 工作流

```mermaid
flowchart LR
    A[图片 / 视频] --> B[抽帧与去重]
    B --> C[人工标注 / API 预标注]
    C --> D[浏览器复核]
    D --> E[YOLO 四类数据集]
    E --> F[训练与误判分析]
    F --> G[推理 / Core ML 导出]
```

| 类别 ID | 标签 | 含义 |
| --- | --- | --- |
| 0 | `pedestrian_signal_red` | 行人红灯 |
| 1 | `pedestrian_signal_green` | 行人绿灯 |
| 2 | `pedestrian_signal_unknown` | 行人灯状态无法确认 |
| 3 | `vehicle_signal` | 车行 / 自行车信号灯 |

类别定义集中在 [ptl/constants.py](ptl/constants.py)。API 预标注使用短标签别名，导出时仍对应上述四类 ID。

## 快速开始

Python 3.10 或更新版本，在仓库根目录执行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m tools.review.serve_review --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`，选择自己的图片文件夹即可建立人工标注任务。API 预标注为可选功能，需要自行填写兼容接口地址、模型及密钥；配置保存在本机，不应提交到仓库。

API 预标注候选框格式（左上角为原点，坐标范围 0–1000）：

```json
{"boxes": [{"label": "pedestrian_red", "bbox": [100, 120, 220, 360], "confidence": 0.95}]}
```

`label` 可取 `pedestrian_red`、`pedestrian_green`、`pedestrian_unknown`、`vehicle_signal`；没有目标时返回 `{"boxes": []}`。

## 训练与评估

准备自己的 YOLO 数据集，复制 [configs/data.example.yaml](configs/data.example.yaml)，修改 `path` 为数据集根目录。图片和同名标签分别放在 `images/{train,val,test}` 与 `labels/{train,val,test}`。

```bash
python -m pip install -e ".[train]"
ptl-train --data configs/data.example.yaml --device cpu --epochs 1 --project runs/detect --name train
# GPU 训练可改为 --device 0，并自行设置训练轮数。
ptl-eval-safety --model runs/detect/train/weights/best.pt --data configs/data.example.yaml --report-only --out outputs/evaluation.md
ptl-predict --model runs/detect/train/weights/best.pt --source data/example/images/test --out outputs/predictions
```

训练输出目录以实际运行日志为准。Ultralytics 可能下载初始权重；数据集和训练权重均需自行准备。评估脚本支持红绿灯混淆分析，正式数据质量检查与报告模式的差别见 `ptl-eval-safety --help`。

## 工具与代码

| 目录 / 脚本 | 用途 |
| --- | --- |
| `ptl/` | 类别约定、标签 IO、匹配与误判统计 |
| `tools/review/` | 浏览器标注、复核、任务分段与进度保存 |
| `scripts/video_to_dataset.py` | 抽帧、感知哈希去重和可选预标注 |
| `scripts/check_leakage.py` | 数据集划分重复检查 |
| `scripts/eval_safety.py` | 红绿灯误判与漏检分析 |
| `scripts/export_coreml.py` | Core ML 模型导出与标签校验 |
| `tests/` | 使用临时样例和 mock 的自动化测试 |

各脚本可通过 `python -m scripts.<脚本名> --help` 查看参数。Core ML 导出需要额外安装 `.[train,export-coreml]`，并在兼容的 macOS / coremltools 环境验证产物。

## 发布范围

本仓库提供工具源码、配置模板和测试，不包含采集视频、训练数据、模型权重、密钥或现场识别效果承诺。自动化测试不等于真实场景精度验证；该工具用于研究和评估，不能据此决定是否过街。

原创工具代码采用 [MIT License](LICENSE)。Ultralytics、Core ML 工具及模型 / 数据集分别遵循其自身条款，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
