"""What ONNX export and 320-downscaling cost, measured on the same held-out test split.

The ONNX legs run on CPU on purpose: that is what the phone will do, and it keeps
Ultralytics from trying to pull onnxruntime-gpu into a read-only Python install.
"""
import json

# Measured already in the first pass; kept so a rerun of the ONNX legs is cheap.
PYTORCH = {"mAP50": 0.907, "mAP50_95": 0.5501, "precision": 0.8906, "recall": 0.8518,
           "ap50_per_class": {"hardhat": 0.9163, "no-hardhat": 0.8976}}


def main():
    from ultralytics import YOLO
    out = {"pytorch-512": PYTORCH}
    print("pytorch-512", PYTORCH, flush=True)
    for tag, path, sz in [("onnx-512", "D:/Portfolio/sitewatch-mobile/model/sitewatch-512.onnx", 512),
                          ("onnx-320", "D:/Portfolio/sitewatch-mobile/model/sitewatch-320.onnx", 320)]:
        m = YOLO(path)
        r = m.val(data="backend/data/hardhat/data.yaml", split="test", imgsz=sz, batch=1,
                  conf=0.001, iou=0.7, workers=0, device="cpu", verbose=False, plots=False)
        out[tag] = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
                    "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                    "ap50_per_class": {m.names[i]: round(float(v), 4) for i, v in zip(r.box.ap_class_index, r.box.ap50)}}
        print(tag, out[tag], flush=True)
    json.dump(out, open("D:/Portfolio/sitewatch-mobile/model/parity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
