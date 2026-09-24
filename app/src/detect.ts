/**
 * The whole detector, on the phone.
 *
 * Sitewatch's server build did this in Python: PIL opened the image, Ultralytics
 * letterboxed it, PyTorch ran the network, and Ultralytics decoded and suppressed the
 * raw output. None of that exists here. What follows is the same four steps written
 * out by hand, because the only thing `onnxruntime-react-native` gives you is a tensor
 * in and a tensor out — every convention Ultralytics quietly applies on both sides has
 * to be reproduced exactly or the boxes land in the wrong place.
 *
 * The two conventions that matter:
 *
 *   1. Letterbox, don't stretch. The model saw letterboxed images in training, so a
 *      stretched one is out of distribution. Scale by the smaller ratio, pad the rest
 *      with grey 114 — Ultralytics' padding value, not an arbitrary one — and centre it.
 *      Then undo exactly that transform on the way out.
 *
 *   2. The output is in input-pixel space, channel-major. Ultralytics exports YOLO11
 *      detect heads as [1, 4 + nc, N]: four box rows then one row per class, already
 *      through sigmoid, with cx/cy/w/h measured in pixels of the 320x320 input rather
 *      than normalised. Reading it as row-major, or assuming 0..1, gives boxes that
 *      look plausible and are wrong.
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';

/**
 * Square input the exported graph expects; the export is fixed-shape.
 *
 * 512 rather than 320, and that was measured rather than assumed. Both sizes were
 * exported and scored on the full held-out test split: 320 costs 6.7 mAP50 points
 * (0.840 against 0.904) and — the number that actually decides it — drops `no-hardhat`
 * AP50 from 0.897 to 0.838. That is roughly one violation in twenty going unflagged,
 * bought in exchange for about 7 ms per photo on a desktop CPU. For an app that scores
 * one photo at a time on a button press, nobody notices the milliseconds and somebody
 * eventually notices the missed violation. 320 remains exported in `model/` for a
 * future live-video mode, where the trade would be a genuine one.
 */
export const INPUT_SIZE = 512;

/** Ultralytics' letterbox fill. Matching it keeps padded borders in distribution. */
const PAD_VALUE = 114;

/** Class order is baked into the weights: index 0 is a worker wearing a hard hat. */
export const CLASSES = ['hardhat', 'no-hardhat'] as const;
export type ClassName = (typeof CLASSES)[number];

/**
 * Sitewatch's shipped operating point, chosen by F1 on `no-hardhat` alone rather than
 * by mAP — see the model card. Carried over unchanged so the phone and the server
 * disagree about nothing except where the arithmetic happens.
 */
export const DEFAULT_CONF = 0.35;
export const DEFAULT_IOU = 0.45;

export interface Detection {
  /** Corners in the *original* image's pixel space, letterboxing already undone. */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: ClassName;
}

export interface RGBAImage {
  width: number;
  height: number;
  /** Row-major RGBA, 4 bytes per pixel — what jpeg-js hands back. */
  data: Uint8Array;
}

/** How a source image was fitted into the square input, kept so it can be reversed. */
export interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
}

/**
 * Resize-with-padding into the NCHW float tensor the graph wants.
 *
 * Returns the transform alongside the tensor: the caller needs it to map boxes back,
 * and deriving it twice is how the two copies drift apart.
 */
export function preprocess(img: RGBAImage): { data: Float32Array; box: Letterbox } {
  const S = INPUT_SIZE;
  const scale = Math.min(S / img.width, S / img.height);
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const padX = Math.floor((S - w) / 2);
  const padY = Math.floor((S - h) / 2);

  // Planar RGB, pre-filled with the pad value so only the image area needs writing.
  const out = new Float32Array(3 * S * S).fill(PAD_VALUE / 255);
  const plane = S * S;

  for (let y = 0; y < h; y++) {
    // Nearest neighbour: the source is already close to 320 after the native resize
    // in imageToRGBA, so interpolating here would cost milliseconds to change almost
    // nothing. The expensive resampling happens once, natively, upstream.
    const sy = Math.min(img.height - 1, Math.floor(y / scale));
    const dstRow = (y + padY) * S + padX;
    const srcRow = sy * img.width;
    for (let x = 0; x < w; x++) {
      const sx = Math.min(img.width - 1, Math.floor(x / scale));
      const s = (srcRow + sx) * 4;
      const d = dstRow + x;
      out[d] = img.data[s] / 255;
      out[plane + d] = img.data[s + 1] / 255;
      out[2 * plane + d] = img.data[s + 2] / 255;
    }
  }
  return { data: out, box: { scale, padX, padY } };
}

/**
 * Turn the raw [1, 4 + nc, N] head into boxes in the original image's coordinates.
 *
 * Filtering by confidence before the coordinate arithmetic matters more than it looks:
 * at 320 there are 2,100 anchors and all but a handful are below threshold, so the
 * early `continue` is the difference between 2,100 box conversions and about twenty.
 */
export function decode(
  raw: Float32Array,
  n: number,
  box: Letterbox,
  imgW: number,
  imgH: number,
  conf: number,
): Detection[] {
  const out: Detection[] = [];
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestCls = 0;
    for (let c = 0; c < CLASSES.length; c++) {
      const s = raw[(4 + c) * n + i];
      if (s > best) {
        best = s;
        bestCls = c;
      }
    }
    if (best < conf) continue;

    const cx = raw[i];
    const cy = raw[n + i];
    const w = raw[2 * n + i];
    const h = raw[3 * n + i];

    // Undo the letterbox: drop the padding, then divide out the scale.
    const x1 = (cx - w / 2 - box.padX) / box.scale;
    const y1 = (cy - h / 2 - box.padY) / box.scale;
    const x2 = (cx + w / 2 - box.padX) / box.scale;
    const y2 = (cy + h / 2 - box.padY) / box.scale;

    out.push({
      // A box straddling the pad border can land outside the photo; clamp so the
      // overlay never draws off-image.
      x1: Math.max(0, Math.min(imgW, x1)),
      y1: Math.max(0, Math.min(imgH, y1)),
      x2: Math.max(0, Math.min(imgW, x2)),
      y2: Math.max(0, Math.min(imgH, y2)),
      score: best,
      cls: CLASSES[bestCls],
    });
  }
  return out;
}

function iou(a: Detection, b: Detection): number {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/**
 * Greedy non-maximum suppression, per class.
 *
 * Per class, not across classes, and deliberately: a head detected as both `hardhat`
 * and `no-hardhat` is a disagreement worth seeing, and cross-class suppression would
 * silently resolve it by confidence. Ultralytics defaults the same way.
 */
export function nms(dets: Detection[], iouThresh = DEFAULT_IOU): Detection[] {
  const kept: Detection[] = [];
  for (const cls of CLASSES) {
    const pool = dets.filter((d) => d.cls === cls).sort((a, b) => b.score - a.score);
    while (pool.length) {
      const top = pool.shift()!;
      kept.push(top);
      for (let i = pool.length - 1; i >= 0; i--) {
        if (iou(top, pool[i]) > iouThresh) pool.splice(i, 1);
      }
    }
  }
  return kept.sort((a, b) => b.score - a.score);
}

/** One loaded graph, reused across photos — creating a session costs ~1 s. */
let session: InferenceSession | null = null;

export async function loadModel(uri: string): Promise<void> {
  if (!session) session = await InferenceSession.create(uri);
}

export function isLoaded(): boolean {
  return session !== null;
}

export interface Result {
  detections: Detection[];
  /** Inference only — preprocessing and JPEG decode are timed separately by the caller. */
  inferenceMs: number;
  /**
   * The undecoded head, plus the transform needed to read it. Returned so that moving
   * the confidence threshold can re-run `decode` and `nms` over the same forward pass
   * instead of paying for another one — the slider is meant to feel free.
   */
  raw: Float32Array;
  anchors: number;
  box: Letterbox;
}

export async function detect(
  img: RGBAImage,
  conf = DEFAULT_CONF,
  iouThresh = DEFAULT_IOU,
): Promise<Result> {
  if (!session) throw new Error('Model not loaded — call loadModel() first.');

  const { data, box } = preprocess(img);
  const input = new Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  const t0 = Date.now();
  const output = await session.run({ [session.inputNames[0]]: input });
  const inferenceMs = Date.now() - t0;

  const head = output[session.outputNames[0]];
  const n = head.dims[2];
  const raw = head.data as Float32Array;

  return {
    detections: nms(decode(raw, n, box, img.width, img.height, conf), iouThresh),
    inferenceMs,
    raw,
    anchors: n,
    box,
  };
}

/** What the compliance readout on screen is counting. */
export function summarise(dets: Detection[]) {
  const compliant = dets.filter((d) => d.cls === 'hardhat').length;
  const violations = dets.filter((d) => d.cls === 'no-hardhat').length;
  const people = compliant + violations;
  return {
    people,
    compliant,
    violations,
    // Undefined rather than 100% when nobody is in frame: an empty photo is not a
    // compliant site, and rendering it as one would be a lie the UI tells by default.
    compliancePct: people ? Math.round((100 * compliant) / people) : null,
  };
}
