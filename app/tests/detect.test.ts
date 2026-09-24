/**
 * The parts of the detector that are pure arithmetic, tested without a phone.
 *
 * Everything in `detect.ts` except the ORT call is deterministic maths reproducing a
 * convention defined elsewhere, which is precisely the kind of code that is wrong in a
 * way that still looks like it works — boxes appear, they are just shifted. So the
 * round-trip cases below build a head by hand from a known box and assert the same box
 * comes back out.
 */

import {
  CLASSES,
  Detection,
  INPUT_SIZE,
  decode,
  nms,
  preprocess,
  summarise,
} from '../src/detect';

/** A flat image of one colour, so preprocessing has a value to check against. */
function solid(width: number, height: number, r: number, g: number, b: number) {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

/** Build the [1, 4 + nc, N] head Ultralytics emits, channel-major, from box rows. */
function head(boxes: Array<{ cx: number; cy: number; w: number; h: number; scores: number[] }>) {
  const n = boxes.length;
  const raw = new Float32Array((4 + CLASSES.length) * n);
  boxes.forEach((b, i) => {
    raw[i] = b.cx;
    raw[n + i] = b.cy;
    raw[2 * n + i] = b.w;
    raw[3 * n + i] = b.h;
    b.scores.forEach((s, c) => {
      raw[(4 + c) * n + i] = s;
    });
  });
  return { raw, n };
}

describe('preprocess', () => {
  it('produces a planar NCHW buffer of the declared size', () => {
    const { data } = preprocess(solid(100, 50, 255, 0, 0));
    expect(data.length).toBe(3 * INPUT_SIZE * INPUT_SIZE);
  });

  it('letterboxes a wide image with grey bars rather than stretching it', () => {
    // 640x320 scales by 0.5 to 320x160, leaving 80px of padding top and bottom.
    const { box, data } = preprocess(solid(640, 320, 255, 0, 0));
    expect(box.scale).toBeCloseTo(0.5);
    expect(box.padX).toBe(0);
    expect(box.padY).toBe(80);

    const plane = INPUT_SIZE * INPUT_SIZE;
    const topBar = 10 * INPUT_SIZE + 10; // inside the padding
    const middle = 160 * INPUT_SIZE + 160; // inside the image
    expect(data[topBar]).toBeCloseTo(114 / 255);
    expect(data[middle]).toBeCloseTo(1); // red channel of a pure-red source
    expect(data[plane + middle]).toBeCloseTo(0); // green
  });

  it('does not pad a square image', () => {
    const { box } = preprocess(solid(200, 200, 0, 0, 0));
    expect(box.padX).toBe(0);
    expect(box.padY).toBe(0);
  });
});

describe('decode', () => {
  it('reads the head channel-major and converts centre-form to corners', () => {
    // One anchor, a 40x40 box centred at (160, 160) of the 320 input, no padding.
    const { raw, n } = head([{ cx: 160, cy: 160, w: 40, h: 40, scores: [0.9, 0.1] }]);
    const out = decode(raw, n, { scale: 1, padX: 0, padY: 0 }, 320, 320, 0.35);

    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ x1: 140, y1: 140, x2: 180, y2: 180, cls: 'hardhat' });
    expect(out[0].score).toBeCloseTo(0.9);
  });

  it('undoes the letterbox so boxes land in source coordinates', () => {
    // The transform a 640x320 source gets: half scale, 80px of vertical padding.
    // A box at the input's centre must come back at the source's centre, (320, 160).
    const { raw, n } = head([{ cx: 160, cy: 160, w: 50, h: 50, scores: [0.1, 0.8] }]);
    const out = decode(raw, n, { scale: 0.5, padX: 0, padY: 80 }, 640, 320, 0.35);

    const d = out[0];
    expect((d.x1 + d.x2) / 2).toBeCloseTo(320);
    expect((d.y1 + d.y2) / 2).toBeCloseTo(160);
    expect(d.x2 - d.x1).toBeCloseTo(100); // 50px at half scale is 100 in the source
    expect(d.cls).toBe('no-hardhat');
  });

  it('takes the highest-scoring class, not the first over threshold', () => {
    const { raw, n } = head([{ cx: 50, cy: 50, w: 10, h: 10, scores: [0.4, 0.95] }]);
    const out = decode(raw, n, { scale: 1, padX: 0, padY: 0 }, 320, 320, 0.35);
    expect(out[0].cls).toBe('no-hardhat');
  });

  it('drops anchors below the threshold', () => {
    const { raw, n } = head([
      { cx: 50, cy: 50, w: 10, h: 10, scores: [0.2, 0.1] },
      { cx: 90, cy: 90, w: 10, h: 10, scores: [0.6, 0.0] },
    ]);
    expect(decode(raw, n, { scale: 1, padX: 0, padY: 0 }, 320, 320, 0.35)).toHaveLength(1);
  });

  it('clamps a box that overhangs the padding back onto the image', () => {
    // Centred on the very top edge, so half the box sits in the pad region.
    const { raw, n } = head([{ cx: 160, cy: 0, w: 40, h: 40, scores: [0.9, 0] }]);
    const out = decode(raw, n, { scale: 1, padX: 0, padY: 0 }, 320, 320, 0.35);
    expect(out[0].y1).toBe(0);
  });
});

describe('nms', () => {
  const box = (x: number, score: number, cls: Detection['cls']): Detection => ({
    x1: x,
    y1: 0,
    x2: x + 100,
    y2: 100,
    score,
    cls,
  });

  it('keeps the highest-scoring box among overlapping duplicates', () => {
    const out = nms([box(0, 0.6, 'hardhat'), box(10, 0.9, 'hardhat')], 0.45);
    expect(out).toHaveLength(1);
    expect(out[0].score).toBeCloseTo(0.9);
  });

  it('keeps boxes that do not overlap', () => {
    expect(nms([box(0, 0.9, 'hardhat'), box(500, 0.8, 'hardhat')], 0.45)).toHaveLength(2);
  });

  it('suppresses within a class but not across them', () => {
    // A head called both hardhat and no-hardhat is a disagreement the UI should show,
    // not something suppression quietly resolves by confidence.
    const out = nms([box(0, 0.9, 'hardhat'), box(0, 0.7, 'no-hardhat')], 0.45);
    expect(out).toHaveLength(2);
  });

  it('returns results in descending confidence', () => {
    const out = nms([box(0, 0.5, 'hardhat'), box(500, 0.95, 'hardhat'), box(1000, 0.7, 'no-hardhat')], 0.45);
    expect(out.map((d) => d.score)).toEqual([0.95, 0.7, 0.5]);
  });
});

describe('summarise', () => {
  const d = (cls: Detection['cls']): Detection => ({ x1: 0, y1: 0, x2: 1, y2: 1, score: 0.9, cls });

  it('counts people as the sum of both classes', () => {
    const s = summarise([d('hardhat'), d('hardhat'), d('no-hardhat')]);
    expect(s).toMatchObject({ people: 3, compliant: 2, violations: 1, compliancePct: 67 });
  });

  it('reports no compliance figure when nobody is in frame', () => {
    // An empty photo is not a compliant site. Showing 100% would be a lie by default.
    expect(summarise([]).compliancePct).toBeNull();
  });
});
