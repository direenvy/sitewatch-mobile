/**
 * Getting raw pixels out of a photo, which React Native does not offer.
 *
 * There is no `getImageData` here: no canvas, no Image.pixels, nothing that hands back
 * bytes. The route that works is to make the image small natively, ask for it as
 * base64, and decode the JPEG in JavaScript.
 *
 * That sounds slow, and would be at full resolution — a 12-megapixel phone photo is
 * ~36 MB of RGBA and several seconds of pure-JS decoding. The fix is ordering: resize
 * *first*, in `expo-image-manipulator`, which is native code. By the time jpeg-js sees
 * the bytes the image is at most 320 on its long side, about 100k pixels, and the
 * decode is a few tens of milliseconds. The expensive resampling happens once, in C.
 */

import * as ImageManipulator from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import { Buffer } from 'buffer';

import { INPUT_SIZE, RGBAImage } from './detect';

export interface LoadedImage {
  /** Downscaled pixels, ready for the detector. */
  small: RGBAImage;
  /** The photo's true dimensions, so boxes can be drawn against the full-size preview. */
  width: number;
  height: number;
  msDecode: number;
}

/**
 * Downscale to the model's input size and decode to RGBA.
 *
 * Quality 1.0 on the intermediate JPEG is not vanity: this file exists only to be
 * turned into a tensor, and compression artefacts at this size are exactly the kind
 * of high-frequency noise a detector picks up on. The file never touches disk for
 * long and is never shown to anyone.
 */
export async function imageToRGBA(uri: string, width: number, height: number): Promise<LoadedImage> {
  const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height, 1);
  const target = { width: Math.round(width * scale), height: Math.round(height * scale) };

  const resized = await ImageManipulator.manipulateAsync(uri, [{ resize: target }], {
    base64: true,
    compress: 1,
    format: ImageManipulator.SaveFormat.JPEG,
  });

  const t0 = Date.now();
  const bytes = Buffer.from(resized.base64!, 'base64');
  // useTArray keeps the result as a Uint8Array instead of a Node Buffer polyfill,
  // which is both smaller and what the preprocessing loop indexes into.
  const raw = decodeJpeg(bytes, { useTArray: true });
  const msDecode = Date.now() - t0;

  return {
    small: { width: raw.width, height: raw.height, data: raw.data },
    width,
    height,
    msDecode,
  };
}
