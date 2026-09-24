"""Does the hand-written TypeScript decoder agree with Ultralytics?

The app reimplements letterboxing, head decoding and NMS in TypeScript, because
onnxruntime-react-native gives you a tensor and nothing else. That code cannot run on
this machine, but its *arithmetic* can: what follows is a line-for-line port of
app/src/detect.ts into Python. If the port agrees with Ultralytics' own predictions on
real photos, the conventions are right and only the JS syntax is untested.

Disagreement here means the phone would draw boxes in the wrong place.
"""
import glob
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

SIZE = 512
PAD = 114
CLASSES = ["hardhat", "no-hardhat"]
CONF, IOU = 0.35, 0.45
ONNX = "D:/Portfolio/sitewatch-mobile/model/sitewatch-512.onnx"


def native_resize(pil):
    """Stand-in for expo-image-manipulator.

    The app never hands a full-resolution photo to the TypeScript loop: it resizes
    natively first, so the long side is already SIZE and the loop that follows is
    effectively a copy. Reproducing that ordering here matters — comparing a
    nearest-neighbour downscale of a 4000px photo against Ultralytics' bilinear one
    measures PIL against OpenCV, not the decoder.
    """
    w, h = pil.size
    k = min(SIZE / w, SIZE / h, 1)
    return pil.resize((round(w * k), round(h * k)), Image.BILINEAR)


def preprocess(img):
    """Port of preprocess(): letterbox into a planar NCHW float buffer."""
    h, w = img.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    nw, nh = round(w * scale), round(h * scale)
    pad_x, pad_y = (SIZE - nw) // 2, (SIZE - nh) // 2

    out = np.full((3, SIZE, SIZE), PAD / 255, dtype=np.float32)
    # Nearest neighbour, exactly as the TS does it.
    ys = np.minimum(h - 1, (np.arange(nh) / scale).astype(int))
    xs = np.minimum(w - 1, (np.arange(nw) / scale).astype(int))
    patch = img[ys][:, xs].astype(np.float32) / 255
    out[:, pad_y:pad_y + nh, pad_x:pad_x + nw] = patch.transpose(2, 0, 1)
    return out[None], (scale, pad_x, pad_y)


def decode(raw, box, iw, ih, conf):
    """Port of decode(): channel-major read, centre-form to corners, unletterbox."""
    scale, pad_x, pad_y = box
    raw = raw[0]                       # (4 + nc, N)
    scores = raw[4:]                   # (nc, N)
    best = scores.max(axis=0)
    cls = scores.argmax(axis=0)
    keep = best >= conf

    cx, cy, bw, bh = raw[0][keep], raw[1][keep], raw[2][keep], raw[3][keep]
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale
    return [
        dict(x1=float(np.clip(a, 0, iw)), y1=float(np.clip(b, 0, ih)),
             x2=float(np.clip(c, 0, iw)), y2=float(np.clip(d, 0, ih)),
             score=float(s), cls=CLASSES[k])
        for a, b, c, d, s, k in zip(x1, y1, x2, y2, best[keep], cls[keep])
    ]


def iou(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    bb = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (aa + bb - inter)


def nms(dets, thresh=IOU):
    """Port of nms(): greedy, per class, never across classes."""
    kept = []
    for c in CLASSES:
        pool = sorted([d for d in dets if d["cls"] == c], key=lambda d: -d["score"])
        while pool:
            top = pool.pop(0)
            kept.append(top)
            pool = [d for d in pool if iou(top, d) <= thresh]
    return sorted(kept, key=lambda d: -d["score"])


def match(mine, theirs):
    """Greedy one-to-one pairing by IoU, so a missing box shows up as unmatched."""
    used, pairs = set(), []
    for i, m in enumerate(mine):
        best, bj = 0.0, None
        for j, t in enumerate(theirs):
            if j in used or t["cls"] != m["cls"]:
                continue
            v = iou(m, t)
            if v > best:
                best, bj = v, j
        if bj is not None and best > 0.5:
            used.add(bj)
            pairs.append((m, theirs[bj], best))
    return pairs, len(mine) - len(pairs), len(theirs) - len(pairs)


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    ref = YOLO(ONNX, task="detect")
    files = sorted(glob.glob("backend/data/hardhat/test/images/*.jpg"))[:40]
    if not files:
        sys.exit("no test images found — run from D:/Portfolio/sitewatch")

    tot_mine = tot_theirs = tot_pairs = 0
    worst_iou, worst_score = 1.0, 0.0
    for f in files:
        pil = Image.open(f).convert("RGB")
        small = np.array(native_resize(pil))
        tensor, box = preprocess(small)
        raw = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
        # Boxes come back in the downscaled image's space, exactly as in the app; scale
        # them up to the original so they are comparable with Ultralytics' output.
        k = pil.size[0] / small.shape[1]
        mine = nms(decode(raw, box, small.shape[1], small.shape[0], CONF))
        for d in mine:
            for key in ("x1", "y1", "x2", "y2"):
                d[key] *= k

        r = ref.predict(f, imgsz=SIZE, conf=CONF, iou=IOU, verbose=False, device="cpu")[0]
        theirs = [
            dict(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                 score=float(s), cls=CLASSES[int(c)])
            for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
        ]

        pairs, extra, missing = match(mine, theirs)
        tot_mine += len(mine)
        tot_theirs += len(theirs)
        tot_pairs += len(pairs)
        for m, t, v in pairs:
            worst_iou = min(worst_iou, v)
            worst_score = max(worst_score, abs(m["score"] - t["score"]))
        if extra or missing:
            # Print the unmatched boxes' confidences. If a disagreement is resampling
            # noise, the odd box sits right against CONF and would have been in or out
            # on a rounding error. If it sits well above, something is actually wrong.
            paired_m = {id(m) for m, _, _ in pairs}
            paired_t = {id(t) for _, t, _ in pairs}
            # Also report each orphan's best IoU against the other side. A genuinely
            # missing detection has no counterpart at all; a box that merely regressed
            # slightly differently still overlaps something.
            odd = [f"port {d['score']:.3f} (best IoU vs ultra {max([iou(d, t) for t in theirs], default=0):.2f})"
                   for d in mine if id(d) not in paired_m]
            odd += [f"ultra {d['score']:.3f} (best IoU vs port {max([iou(d, m) for m in mine], default=0):.2f})"
                    for d in theirs if id(d) not in paired_t]
            print(f"  mine={len(mine)} ultralytics={len(theirs)}  unmatched: {', '.join(odd)}")

    print(f"\nimages            {len(files)}")
    print(f"boxes (port)      {tot_mine}")
    print(f"boxes (ultralytics) {tot_theirs}")
    print(f"matched           {tot_pairs}")
    print(f"worst IoU on a matched pair   {worst_iou:.4f}")
    print(f"largest score disagreement    {worst_score:.4f}")
    # Exact agreement is not the right bar and never was. The port resamples with
    # PIL where Ultralytics uses OpenCV, so scores differ in the third decimal, and
    # any detection sitting within that margin of the threshold can fall either side.
    # What would indicate a *convention* bug is systematically shifted boxes — so the
    # bar is that essentially every box pairs up, and paired boxes sit almost exactly
    # on top of each other.
    rate = tot_pairs / max(tot_theirs, 1)
    ok = rate >= 0.97 and worst_iou > 0.85 and worst_score < 0.05
    print(f"\nagreement         {rate:.1%} of Ultralytics' boxes")
    print("VERDICT:", "agree, to within resampling noise" if ok else "DISAGREE")


if __name__ == "__main__":
    main()
