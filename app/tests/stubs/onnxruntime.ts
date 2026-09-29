/**
 * Stand-in for `onnxruntime-react-native` under Jest.
 *
 * The unit tests cover the arithmetic around the model — letterboxing, the
 * channel-major head read, coordinate round-trips, suppression — none of which calls
 * the runtime. Importing the real package pulls in React Native's Flow-typed source,
 * which Node cannot parse, so the tests would fail for a reason that has nothing to do
 * with the code under test.
 *
 * Inference itself is verified elsewhere and more convincingly: `model/crosscheck.py`
 * runs a line-for-line port of `detect.ts` against Ultralytics' own predictions on real
 * photos. A mock here could never have caught a misread convention anyway.
 */
export class InferenceSession {
  static async create(): Promise<never> {
    throw new Error('InferenceSession is stubbed in tests; inference is covered by model/crosscheck.py');
  }
}

export class Tensor {
  constructor(
    public type: string,
    public data: Float32Array,
    public dims: number[],
  ) {}
}
