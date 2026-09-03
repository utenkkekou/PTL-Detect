#!/usr/bin/env python3
"""T8 — 导出 4 类信号灯检测器为 Core ML(iOS/Vision)。真正 N×4, 非 N×80。

⚠️ 必须在 macOS(首选) 或 x86_64 Linux 上运行(Windows 被 exporter 硬阻断)。

【关键修复, 经源码级核实(ultralytics 8.4.x exporter.py) + 对抗核实(high confidence)】
现场发现 format="coreml"/"mlpackage" 走 MLProgram 路径, 对 nc%80!=0 会把类别 pad 到 80
(exporter.py:1474, 是 issue #22309 的 workaround, 仅 MLProgram), 导致 confidence=N×80、
classLabels 带数字占位类 —— 违反 App 的精确类别契约, 被 fail-closed 拒载。
改用 **format="mlmodel"(旧版 NeuralNetwork 路径)**: 不 pad、输出 N×4、恰好4类、仍加 NMS,
经 VNCoreMLRequest 出 [VNRecognizedObjectObservation]。原始契约本就允许 .mlmodel。
  - :918 mlmodel=(format=="mlmodel"); :974 convert_to="neuralnetwork"; :1474 pad跳过(mlprogram=False)
  - :1343 输出(anchors,nc)+(anchors,4); :1391 stringClassLabels=4个真实类
说明: .mlmodel 是 deprecated-but-supported 的合法 Vision 对象检测格式; FP16 走旧 quantize 路径。
ANE 驻留需在 Mac 上真机 Performance Report 确认(NeuralNetwork 可上 ANE, 但非最新 MLProgram 路径)。

依赖(导出机): ultralytics + coremltools>=9.0 + "numpy<=2.3.5"
产物: PedestrianSignalIconDetector.mlmodel -> (Mac)xcrun coremlc compile -> .mlmodelc

用法:
  python scripts/export_coreml.py --weights best.pt --imgsz 960 --iou 0.45 --conf 0.25 --outdir models
"""
import argparse
import platform
import sys
from pathlib import Path

EXPECTED_NAMES = {
    0: "pedestrian_signal_red",
    1: "pedestrian_signal_green",
    2: "pedestrian_signal_unknown",
    3: "vehicle_signal",
}
FINAL_STEM = "PedestrianSignalIconDetector"


def preflight():
    if platform.system() == "Windows":
        sys.exit("\n[STOP] Core ML 导出在 Windows 上不支持。拷 best.pt 到 macOS(首选)/x86_64 Linux 再跑。\n"
                 "依赖: pip install ultralytics 'coremltools>=9.0' 'numpy<=2.3.5'\n")


def export_from_pt(pt, imgsz, iou, conf, half):
    from ultralytics import YOLO
    m = YOLO(str(pt))
    names = {int(k): v for k, v in m.names.items()}
    print("[info] model.names =", names)
    assert names == EXPECTED_NAMES, (
        f"model.names {names} != 契约 {EXPECTED_NAMES}. 顺序即索引, 不符会导致 Vision 标签错位; "
        "先修 data.yaml 的 names 顺序再训/导。")
    # KEY: format="mlmodel"(NeuralNetwork), 不是 "coreml"/"mlpackage"(MLProgram会pad到80)
    out = m.export(format="mlmodel", nms=True, imgsz=imgsz, iou=iou, conf=conf, half=half)
    return Path(out)


def relabel_and_save(mlmodel_path, out_path):
    import coremltools as ct
    spec = ct.utils.load_spec(str(mlmodel_path))
    assert spec.WhichOneof("Type") == "pipeline", "期望 pipeline(主干->NMS)"
    nms_model = spec.pipeline.models[-1]
    assert nms_model.HasField("nonMaximumSuppression"), "末端不是 NonMaximumSuppression"
    print("trained labels(顺序):", list(nms_model.nonMaximumSuppression.stringClassLabels.vector))
    # Xcode 拖拽预览提示(不影响运行时)
    spec.description.metadata.userDefined["com.apple.coreml.model.preview.type"] = "objectDetector"
    ct.models.MLModel(spec).save(str(out_path))
    return out_path


def _last_dim(feat):
    ma = feat.type.multiArrayType
    sr = ma.shapeRange.sizeRanges
    if len(sr) >= 2:
        return sr[-1].lowerBound, sr[-1].upperBound
    if len(ma.shape) >= 2:
        return ma.shape[-1], ma.shape[-1]
    return None, None


def verify(mlmodel_path, expected_names=EXPECTED_NAMES):
    """硬契约校验: 末端NMS / 精确标签顺序 / confidence N×nc / coordinates N×4"""
    import coremltools as ct
    ordered_names = [expected_names[i] for i in range(len(expected_names))]
    nc = len(ordered_names)
    spec = ct.utils.load_spec(str(mlmodel_path))
    ok = spec.WhichOneof("Type") == "pipeline"
    if not ok:
        print("FAIL: 不是 pipeline"); return False
    nms = spec.pipeline.models[-1]
    if not nms.HasField("nonMaximumSuppression"):
        print("FAIL: 末端非NMS -> Vision只会给 VNCoreMLFeatureValueObservation"); ok = False
    else:
        labels = list(nms.nonMaximumSuppression.stringClassLabels.vector)
        print(f"classLabels({len(labels)}):", labels)
        if labels != ordered_names:
            print(f"FAIL: 标签名称/顺序不符, 期望 {ordered_names}"); ok = False
    outs = {o.name: o for o in spec.description.output}
    for name, want in (("confidence", nc), ("coordinates", 4)):
        if name not in outs:
            print(f"FAIL: 缺输出 {name}"); ok = False; continue
        lo, hi = _last_dim(outs[name])
        print(f"{name} 末维: [{lo},{hi}]")
        if lo != want or hi != want:
            print(f"FAIL: {name} 末维=[{lo},{hi}], 期望恰好 {want}"); ok = False
    print("VERIFY:", "PASS ✅" if ok else "FAIL ❌")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="训好的 best.pt (或已导出的 .mlmodel 只做relabel+校验)")
    ap.add_argument("--imgsz", type=int, default=960, choices=[640, 960, 1280])
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true", default=True, help="FP16(NeuralNetwork走旧quantize路径)")
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()
    preflight()
    assert args.imgsz % 32 == 0
    import ultralytics
    print(f"[info] ultralytics {ultralytics.__version__}")
    src = Path(args.weights)
    if src.suffix == ".pt":
        exported = export_from_pt(src, args.imgsz, args.iou, args.conf, args.half)
    elif src.suffix == ".mlmodel":
        exported = src
    elif src.suffix == ".mlpackage":
        sys.exit("ERROR: 已 pad 的 .mlpackage 无法可靠改回 N×4(切片会重入#22309坏框)。请用 best.pt 走 mlmodel 路径。")
    else:
        sys.exit(f"不支持的输入: {src}")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{FINAL_STEM}.mlmodel"
    relabel_and_save(Path(exported), final)
    ok = verify(final)
    print(f"\n[OUT] {final.resolve()}")
    print("下一步(Mac): xcrun coremlc compile "
          f"{final} {outdir}  ->  {FINAL_STEM}.mlmodelc")
    print("  然后 Xcode Performance Report 确认 ANE; VNCoreMLRequest 确认返回 [VNRecognizedObjectObservation]")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

# 等价一行(不做relabel/校验):
#   yolo export model=best.pt format=mlmodel nms=True imgsz=960 iou=0.45 conf=0.25 half=True
#   ⚠️ 必须 format=mlmodel(不是 coreml/mlpackage), 否则又 pad 成 N×80。仅 macOS/Linux。
