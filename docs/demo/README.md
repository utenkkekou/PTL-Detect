# 离线标注演示素材

`signals-demo.png` 由仓库内的 `make_demo.py` 绘制，只包含几何信号灯示意，用于体验四类人工标注、复核保存和标签导出。它不是实拍数据、模型推理结果或检测性能证明。

在仓库根目录、已安装基础依赖的环境中运行：

```bash
python docs/demo/make_demo.py
python -m tools.review.serve_review --host 127.0.0.1 --port 8000 --no-browser --config outputs/readme-demo/config.json
```

打开 `http://127.0.0.1:8000`，图片文件夹填 `docs/demo`，任务输出文件夹填 `outputs/readme-demo`，选择“创建空白任务”后进入审核。无需填写 API 密钥。

1. 按 `R`，拖拽框选一个灯头；对其余灯头重复操作。
2. 选中框后按 `1` / `2` / `3` / `4`，分别设置人行红、人行绿、人行不明、车行/自行车。
3. 点击“通过”或按空格提交复核；刷新页面可验证标注已保存。

完成复核后导出 YOLO 标签：

```bash
python -m scripts.ptl_export_labels --images-dir docs/demo --manifest outputs/readme-demo/review_manifest.json --progress outputs/readme-demo/review_progress.json --out outputs/readme-demo/yolo
```

输出包括 `labels/` 下的标签文件和 `ptl_index.csv`，不会复制原图。再次导出前请使用新的输出目录，或按需使用脚本的 `--allow-existing` 选项。

`../assets/review-workbench.png` 是以上流程完成后的真实浏览器截图。图中的四个框为人工标注，界面显示的 `conf 1` 是新建框的默认值，不代表模型的置信度或准确率。脚本的“自动打标准确率”输出也不适用于此空白人工标注演示。
